# Reducing-bias-for-toxicity-classification

## Project Overview:
As Machine Learning is increasingly used in our day to day lives, issues surrounding the reinforcement of existing biases against minority identities are increasingly important. We are exploring this from the scope of online toxic comment classification

Classical ML classification models perform very well in terms of accuracy when identifying toxic comments. However these methods have a tendency to over-weight words that refer to a particular identity (e.g. Black, Muslim, LGBTQ...) leading to non-toxic examples mentioning these identites being classified as toxic (False Positives). This can be somewhat ameliorated via techniques such as hyperparameter optimization but this represents a fundamental issue with these models. 

Our aim is to train a neural network model, primarily an LSTM model, to classify toxic comments. RNN's such as LSTM are optimally suited to tackling this problem due to their ability to parse through sequences, such as a comment. The main idea here is that an RNN should be better at deciphering the underlying *context* of a sequence based on the word-embeddings passed in. 

We compare the results of our LSTM model and a few classical ML models to suggest areas of further improvements.

### Dataset
The dataset used comes from the Kaggle Compeition [Jigsaw Unintended bias in toxicity classification]('https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/description')

As a quick overview, the data contains online comments and a labelled target column indicating whether the comment is toxic. In addition to this, we are provided with identity labels, which show whether a particular comment had a specific mention to a certain identity.

### Evaluation
We will use the standard acccuracy score as a gauge of how good the models are at the main task of identifying toxic comments. However we will also use a specialized metric specified by Jigsaw AI. Please read the [explanation here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation) for more details. 

Effectively we split the predictions into those for each selected subgroup, calculate the AUC and then take the weighted means of the subgroup AUC and overall AUC. A low score here suggests that while your model may be good at identifying toxic comments, it is biased against certain identity groups.

## Project Set up:

- Download project from [github](https://github.com/GovindSuresh/reducing-bias-in-toxicity-classification)

- Project Requirements:
     - Please create a new virtual environment using the ```requirements.txt``` file provided. The project mostly uses standard python ML libraries such as scikit-learn, pandas, and numpy. For NLP work we have primarily used the NLTK library.  For deep learning we have used TensorFlow 2.0 and Keras 2.4.1. 
     
- Data and other downloads:
     - The dataset can be downloaded from [kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). Please download and store this in the subdirectory named ```data``` in your main project directory. If not you will need to manually change certain relative file paths within the code.
     - Word Embeddings can be downloaded from the [GloVE](https://nlp.stanford.edu/projects/glove/) page. We have specifically used the Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) file, however others can be used. Please download and store this file in the subdirectory named ```embeds``` in the main project directory.

## Contents:

### Notebooks:

At present the key files to read are contained in the notebooks folder. These notebooks go through the modelling process for my initial results which are discussed in the `Report.ipynb` file and also in more detail on my [personal website](https://govindsuresh.github.io/articles/2020-01/reducing-bias-in-toxic-comments).
- EDA.ipynb: This notebook goes through the initial cleaning steps on the data, mainly dropping columns that wouldn't be used and also dealing with null values. Some data exploration was also done. 
     
- Preprocessing.ipynb: Here, we develop the NLP process pipelines to prepare our data for the LSTM model and other ML models trained. The two sets of model types required different NLP  techniques given that we were using pre-trained word embeddings for the LSTM model. 
     
- ML_models.ipynb: Here, we go through the usual modelling process for our 'regular' ML classification models. 
     
- NN_model.ipynb: This is the notebook for the LSTM model. 

### Scripts:

*These files are currently being worked on - 22/04/19*


## Next Steps:

We are currently in the process of taking the cleaning and preprocessing steps from the notebooks and converting them into a python script to create a data pipeline. We will then move onto the models to turn them into scripts.

The final aim is to have the project in a form such that one could easily run the scripts to train the model - likely on cloud platforms such as GCP's AI Platform and AWS Sagemaker

*22/04/19*

- ~~Write LSTM preprocessing script~~

*04/05/19*

- Write LSTM training script
- Write LSTM inference script
- Test pipeline end-to-end








 
