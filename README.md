# Knowledge Graph augmented NLP

Source code for the NAACL'18 Long Paper [Learning beyond datasets: Knowledge Graph augmented Neural Networks for Natural Language Processing](http://www.aclweb.org/anthology/N18-1029)


## Installation

### Download

News20: '20news-19997.tar.gz - Original 20 Newsgroups data set', file from the website http://qwone.com/~jason/20Newsgroups/.

SNLI: Download  SNLI 1.0 from the website : https://nlp.stanford.edu/projects/snli/

Place the News20 and SNLI dataset in a folder and update DATASET_PATH variable in the config.py in news20/ and snli/ folder. 

### Setup

In order to setup the project you need to play around with some APIs. Run the setup() in clean_dataset.py  and prepare_dataset.py, both the API calls are commented out. This is a one time thing and it doesn't need to used during training process.

### Run

```
python main.py --arguments
```

Arguments regarding the training process, can be found in source code in the file main.py. Please refer to the paper for exact hyperparameters.

## Libraries

The development was performed using Tensorflow 1.0, whichever was the latest version in May, 2017.<br>
Python version = 2.7

## Citation
```
@inproceedings{k-m-etal-2018-learning,
    title = "Learning beyond Datasets: Knowledge Graph Augmented Neural Networks for Natural Language Processing",
    author = "K M, Annervaz  and
      Basu Roy Chowdhury, Somnath  and
      Dukkipati, Ambedkar",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1029",
    doi = "10.18653/v1/N18-1029",
    pages = "313--322",
    abstract = "Machine Learning has been the quintessential solution for many AI problems, but learning models are heavily dependent on specific training data. Some learning models can be incorporated with prior knowledge using a Bayesian setup, but these learning models do not have the ability to access any organized world knowledge on demand. In this work, we propose to enhance learning models with world knowledge in the form of Knowledge Graph (KG) fact triples for Natural Language Processing (NLP) tasks. Our aim is to develop a deep learning model that can extract relevant prior support facts from knowledge graphs depending on the task using attention mechanism. We introduce a convolution-based model for learning representations of knowledge graph entity and relation clusters in order to reduce the attention space. We show that the proposed method is highly scalable to the amount of prior information that has to be processed and can be applied to any generic NLP task. Using this method we show significant improvement in performance for text classification with 20Newsgroups (News20) {\&} DBPedia datasets, and natural language inference with Stanford Natural Language Inference (SNLI) dataset. We also demonstrate that a deep learning model can be trained with substantially less amount of labeled training data, when it has access to organized world knowledge in the form of a knowledge base.",
}
```