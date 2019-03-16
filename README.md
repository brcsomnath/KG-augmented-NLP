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