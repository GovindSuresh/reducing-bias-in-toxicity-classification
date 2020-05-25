#############################################################################################
#
# LSTM model evaluation script 
# Script used to evaluate models on the test set and also calculate the bias weighted AUC
#
#############################################################################################

# Regular imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
import yaml

# Sklearn imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# TensorFlow imports 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import text, sequence


def text_padder(text, tokenizer):

    return sequence.pad_sequences(tokenizer.texts_to_sequences(text), maxlen=MAX_LEN_SEQ, padding='post')

def model_evaluation(test_preds, test_labels, test_df, model_name):
    '''
    Evaluates the models against accuracy, F1 Score, and the final weighted bias metric as discussed in the project write-up
    Inputs:
    Array: test_preds - Predicted labels for the test set
    Array: test_labels - True labels for the test set 
    Dataframe:  test_df - test dataframe, needed for the final bias metric calculation

    Outputs:
    nn_bias_metrics_df_test = dataframe showing bias metric scores against the specific identify subgroups
    results_df = dataframe of metrics for the model 
        
    '''
    # Define subgroup metrics
    SUBGROUP_AUC = 'subgroup_auc'
    BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
    BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

    # These calculations have been provided by Jigsaw AI for scoring based on the metrics of the kaggle competition
    # https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation

    # They work by filtering the relevant dataframe into specific subgroups and using the roc_auc_score metric from sklearn.

    def compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def compute_subgroup_auc(df, subgroup, label, model_name):
        subgroup_examples = df[df[subgroup]]
        return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

    def compute_bpsn_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df.loc[df[subgroup] & ~df[label]]
        non_subgroup_positive_examples = df.loc[~df[subgroup] & df[label]]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return compute_auc(examples[label], examples[model_name])

    def compute_bnsp_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df.loc[df[subgroup] & df[label]]
        non_subgroup_negative_examples = df.loc[~df[subgroup] & ~df[label]]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return compute_auc(examples[label], examples[model_name])

    def compute_bias_metrics_for_model(dataset,
                                    subgroups,
                                    model,
                                    label_col,
                                    include_asegs=False):
        """Computes per-subgroup metrics for all subgroups and one model."""
        records = []
        for subgroup in subgroups:
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset.loc[dataset[subgroup]])
            }
            record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
            record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
            record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
            records.append(record)
        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

    def calculate_overall_auc(df, model_name):
        true_labels = df[TEST_TARGET_COL]
        predicted_labels = df[model_name]
        return roc_auc_score(true_labels, predicted_labels)

    def power_mean(series, p):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
        bias_score = np.average([
            power_mean(bias_df[SUBGROUP_AUC], POWER),
            power_mean(bias_df[BPSN_AUC], POWER),
            power_mean(bias_df[BNSP_AUC], POWER)
        ])
        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
    
    # Binarize the y_predictions:

    test_preds_bin = np.where(test_preds >=0.5, 1, 0)

    # Compute standard metrics 
    model_accuracy = accuracy_score(test_labels,test_preds_bin)
    model_precision = precision_score(test_labels, test_preds_bin)
    model_recall = recall_score(test_labels,test_preds_bin)
    model_f1 = f1_score(test_labels, test_preds_bin)

    # For the bias metrics we need the additional identity columns from the test_dataframe. Therefore merge
    # Test predictions back to dataframe.

    test_results = test_df.merge(test_preds_bin, how='inner', on='id')

    # Compute the bias metrics dataframe and final weighted score
    nn_bias_metrics_df_test = compute_bias_metrics_for_model(test_results, IDENTITY_COLUMNS, model_name, TEST_TARGET_COL)
    nn_final_metric_test = get_final_metric(nn_bias_metrics_df_test, calculate_overall_auc(test_results, model_name))

    results_df = pd.DataFrame(data=[[model_accuracy, model_precision, model_recall, model_f1, nn_final_metric_test]], index=[model_name], 
                                columns=['model_accuracy','model_precision', 'model_recall', 'model_f1', 'final_weighted_bias'],
                            )

    return nn_bias_metrics_df_test, results_df

# Set up parser
def set_args():
    parser = argparse.ArgumentParser(description='LSTM training script')
    parser.add_argument('test_file', type=str, help='Filepath of the clean test csv file')
    parser.add_argument('tokenizer_path', type=str, help='Filepath of the tokenizer for the model to be evaluated')
    parser.add_argument('model_path', type=str, help='Path to the model .h5 file to load ')
    parser.add_argument('model_name', type=str, help='Name to give the trained model')
    parser.add_argument('yml_filepath', type=str, help='Location of yml file with model parameters')

    return parser.parse_args()

if __name__ == '__main__':

    # Read in args
    ARGS = set_args()
    TEST_FILE = ARGS.test_file
    TOKENIZER_PATH = ARGS.tokenizer_path
    MODEL_PATH = ARGS.model_path
    MODEL_NAME = ARGS.model_name
    YAML_FILE = ARGS.yml_filepath

    # Pulling in relevant yaml arguments
    stream = open(YAML_FILE, 'r')
    param_dict = yaml.load(stream, Loader=yaml.SafeLoader)

    TEST_TEXT_COL = param_dict['TEST_TEXT_COL']
    TEST_TARGET_COL = param_dict['TEST_TARGET_COL']

    print("Loading Model \n")
    # Load in the model
    model = load_model(MODEL_PATH)

    print(model.summary())

    # Load in the test data
    test_df = pd.read_csv(TEST_FILE, index_col=0)

    # Load in the tokenizer for this model
    with open(TOKENIZER_PATH) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    # Preprocess test data
    X_test = text_padder(test_data[TEST_TEXT_COL], tokenizer)
    y_test = np.asarray(test_df[TEST_TARGET_COL])

    # Get model predictions
    y_preds = model.predict(X_test)

    # Calculate scores and generate dataframes
    bias_metrics_df, results_df = model_evaluation(y_preds, test_df[TEST_TARGET_COL], test_df, MODEL_NAME)
        