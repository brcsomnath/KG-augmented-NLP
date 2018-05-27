import tensorflow as tf 
import numpy as np 
import sys
import os
import time

from tensorflow.contrib.rnn.python.ops import rnn, rnn_cell

class SNLI(object):

	"""
	Initializes parameters of the model from the driver file
	Implements the baseline model using LSTM
	Trains the model over specified number of epochs
	"""

	def __init__(self, sess, train_src, train_targ, train_label, test_src, test_targ, test_label, val_src,
				val_targ, val_label, embedding, n_input = 300, n_steps = 86, train=False, n_hidden1 = 200,
				n_hidden2 = 10, n_classes = 3, learning_rate = 0.001, training_epochs = 100, 
				batch_size = 128, vocabulary_size = 36991):

		"""
		Model parameter initialization
		"""
		self.n_input = n_input
		self.n_steps = n_steps
		self.n_hidden1 = n_hidden1
		self.n_hidden2 = n_hidden2
		self.n_classes = n_classes
		self.learning_rate = learning_rate
		self.training_epochs = training_epochs
		self.batch_size = batch_size

		self.sess = sess
		self.vocab_size = vocabulary_size
		self.train = train

		self.train_src = train_src
		self.train_targ = train_targ
		self.train_label = train_label
		self.test_src = test_src
		self.test_targ = test_targ
		self.test_label = test_label
		self.val_src = val_src
		self.val_targ = val_targ
		self.val_label = val_label
		self.embed = embedding

	def length(self, sequence):

		"""
		Used to calculate length of a sequence in a batch of variable length sentences
		Returns length of all sequences in a batch as an 1D tensor
		"""

		used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
		length = tf.reduce_sum(used, reduction_indices=1)
		length = tf.cast(length, tf.int32)
		return length

	def initializer(self):
		return tf.random_normal_initializer(stddev=0.01)

	def weight_variable(self, shape, name):
		"""
		Weight initialization with zero mean gaussian
		"""

	    initial = tf.truncated_normal(shape, stddev = 0.1)
	    return tf.Variable(initial, name=name)

	def bias_variable(self, shape, name):
	    initial = tf.constant(0.1, shape = shape)
	    return tf.Variable(initial, name=name)

	def LSTM(self, input_seq, weight, bias):

		"""
		LSTM module for input sentence sequence
		Processes sentences using dynamic length of sequences
		Returns the mean of all the hidden layers passed through a non-linearity ($c$)
		"""

		with tf.variable_scope("LSTM"):
			X = tf.nn.embedding_lookup(self.embed_init, input_seq)

			cell = tf.contrib.rnn.LSTMCell(self.n_hidden1, state_is_tuple = True, initializer= self.initializer())
			output, state = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32, sequence_length = self.length(X))
			output_all = tf.reduce_mean(output,1)
			output_all = tf.nn.relu(tf.add(tf.matnmul(output_all, weight),bias))
			return output_all

	def classify(self, input_src, input_targ):
		"""
		Receives the input sentence batch
		Acts as a driver function for the LSTM
		Returns the prediction after passing the LSTM vectors through a 1-layer NN
		"""
		with tf.variable_scope("src"):
			W_src = self.weight_variable([self.n_hidden1, self.n_hidden2], name="src_w")
			B_src = self.bias_variable([self.n_hidden2], name="src_b")
			y_src = self.LSTM(input_src, W_src, B_src)

		with tf.variable_scope("targ"):
			W_targ = self.weight_variable([self.n_hidden1, self.n_hidden2], name="targ_w")
			B_targ = self.bias_variable([self.n_hidden2], name="targ_b")
			y_targ = self.LSTM(input_targ, W_targ, B_targ)

		y_all = tf.concat([y_src, y_targ], 1)
		with tf.variable_scope("final"):
			W = self.weight_variable([2*self.n_hidden2, self.n_classes], name="w")
			B = self.bias_variable([self.n_classes], name="b")
			self.y_out = tf.add(tf.matmul(y_all, W), B)

	def run(self):
		"""
		Driver function to train the entire model for SNLI
		"""
		
		self.W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.n_input]), trainable=False, name="W")
		self.embedding_placeholder = tf.placeholder("float", [None, None])
		self.embed_init = self.W.assign(self.embedding_placeholder)

		x1 = tf.placeholder("int64", [None, self.n_steps])
		x2 = tf.placeholder("int64", [None, self.n_steps])
		y = tf.placeholder("int64", [None, self.n_classes])

		self.total_batches = len(self.train_src)//self.batch_size
		self.classify(x1, x2)

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= self.y_out))
		optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
		correct_prediction = tf.equal(tf.argmax(self.y_out,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		if self.train:
			try:
				tf.global_variables_initializer().run()
			except:
				tf.initialize_all_variables().run()
			
			for epoch in range(self.training_epochs):
				epoch_loss = 0
				for batch in range(self.total_batches):
					batch_x1 = self.train_src[batch*self.batch_size: (batch+1)*self.batch_size]
					batch_x2 = self.train_targ[batch*self.batch_size: (batch+1)*self.batch_size]
					batch_y = self.train_label[batch*self.batch_size: (batch+1)*self.batch_size]
					_, c, acc = self.sess.run([optimizer, loss, accuracy],feed_dict={x1: batch_x1, x2: batch_x2, y : batch_y, self.embedding_placeholder: self.embed})
					epoch_loss += c
					print('Batch: %d: Loss = %lf, Mini-batch accuracy: %lf ' %(batch, c, acc*100.0))

				saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)				
				saver.save(self.sess, 'model/SNLI-lstm')

				acc = self.sess.run(accuracy, feed_dict={x1: self.val_src, x2: self.val_targ, y : self.val_label, self.embedding_placeholder: self.embed})
				print('Validation Accuracy: %lf' % (float(acc)*100.0))
				print('Epoch %d: Average Loss: %lf' % (epoch, 1.0*epoch_loss/self.total_batches))
			