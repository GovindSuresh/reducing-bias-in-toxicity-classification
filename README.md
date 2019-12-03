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

## Project Structure
