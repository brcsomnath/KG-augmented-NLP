import tensorflow as tf 

import os
import numpy as np
import pprint

from lstm_model import SNLI
from kg_cnn import KG_SNLI_CNN
import get_dataset


"""
Argument parameters from the user along with their default value
"""

flags = tf.app.flags
flags.DEFINE_string("entity_path", "../../ProjE/results/entity.txt", "Path to the trained entity vectors")
flags.DEFINE_string("relation_path", "../../ProjE/results/relation.txt", "Path to the trained relation vectors")
flags.DEFINE_string("dataset", "snli", "The name of the dataset [news20]")
flags.DEFINE_string("model", "KGlstm", "The name of the classification model [lstm, KGlstm]")
flags.DEFINE_integer("n_input", 300, "Dimension of the input word vectors [1000]")
flags.DEFINE_integer("n_steps", 86, "Maximum Sequence length of the LSTM [300]")
flags.DEFINE_integer("n_hidden1", 200, "Dimension of the hidden layer 1 of LSTM [200]")
flags.DEFINE_integer("n_hidden2", 25, "Dimension of the hidden layer 2 of LSTM [10]")
flags.DEFINE_integer("n_classes", 3, "Number of class of the output [20]")
flags.DEFINE_integer("entity_dim", 50, "Dimension of the entity vectors [50]")
flags.DEFINE_integer("relation_dim", 50, "Dimension of the relation vectors [50]")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate of the Adam Optimizer [0.001]")
flags.DEFINE_integer("training_epochs", 20, "Number of training epochs [20]")
flags.DEFINE_integer("vocabulary_size", 36991, "Vocabulary Size of the corpus [50594]")
flags.DEFINE_integer("batch_size", 2048, "Batch size for training [10]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [True]")
flags.DEFINE_boolean("lstm_out", False, "True for considering only the final state, False for taking all the time steps [True]")

FLAGS = flags.FLAGS

def main(_):

	"""
	Sets the training parameters for GPU
	Retrieves the dataset
	Calls the utility function along with training parameters
	"""

	pp = pprint.PrettyPrinter()
	pp.pprint(flags.FLAGS.__flags)

	run_config = tf.ConfigProto()
 	run_config.gpu_options.allow_growth=True
 	#run_config.gpu_options.per_process_gpu_memory_fraction = 0.5

	with tf.Session(config = run_config) as sess:
		if FLAGS.dataset == 'snli':
			train_src, train_targ, train_label,	test_src, test_targ, test_label, val_src, val_targ, val_label, embedding = get_dataset.get_data2vec('snli/')

		if FLAGS.model == 'lstm':
			model = SNLI(sess, train_src,
							train_targ, train_label, test_src,
						    test_targ, test_label, val_src,
							val_targ, val_label, embedding,
							n_input = FLAGS.n_input,
							n_steps = FLAGS.n_steps,
							train = FLAGS.train,
							n_hidden1 = FLAGS.n_hidden1,
							n_hidden2 = FLAGS.n_hidden2,
							n_classes = FLAGS.n_classes,
							learning_rate = FLAGS.learning_rate,
							training_epochs = FLAGS.training_epochs,
							batch_size = FLAGS.batch_size)

		if FLAGS.model == 'kgcnn':
			model = KG_SNLI_CNN(sess, train_src,
							train_targ, train_label, test_src,
						    test_targ, test_label, val_src,
							val_targ, val_label, embedding,
							n_input = FLAGS.n_input,
							n_steps = FLAGS.n_steps,
							train = FLAGS.train,
							n_hidden1 = FLAGS.n_hidden1,
							n_hidden2 = FLAGS.n_hidden2,
							n_classes = FLAGS.n_classes,
							learning_rate = FLAGS.learning_rate,
							training_epochs = FLAGS.training_epochs,
							batch_size = FLAGS.batch_size)

		model.run()

if __name__=='__main__':
	tf.app.run()