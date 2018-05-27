"""
This file implements the architecture of Convolutional based fact retrieval from Knowledge Graphs
Dataset: 20Newsgroups
Knowledge Graph: Freebase
"""


import tensorflow as tf 
import numpy as np 
import sys
import os
import time
import logging

from tensorflow.contrib.rnn.python.ops import rnn, rnn_cell
import graph_reader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KG_CNN(object):

	"""
	Initializes parameters of the model from the driver file
	Implements the convolutional based architecture using attention mechanism and LSTMs
	Trains the model over specified number of epochs
	"""

	def __init__(self, sess, data, label, embedding, entity_path = 'res_50/entity2vec_cluster.bern', 
				relation_path = 'res_50/relation2vec_cluster.bern', n_input = 300, n_steps = 300, train=False,
				n_hidden1 = 200, n_hidden2=50, n_classes = 20, entity_dim = 50, relation_dim = 50, learning_rate = 0.001, 
				training_epochs = 100, batch_size = 128, n_clusters=20, vocabulary_size = 50594):


		"""
		Model parameter initialization
		"""

		self.entity_path = entity_path
		self.relation_path = relation_path

		self.n_input = n_input
		self.n_steps = n_steps
		self.n_hidden1 = n_hidden1
		self.n_hidden2 = n_hidden2
		self.n_classes = n_classes
		self.n_clusters = n_clusters
		self.entity_dim = entity_dim
		self.relation_dim = relation_dim

		self.learning_rate = learning_rate
		self.training_epochs = training_epochs
		self.batch_size = batch_size
		
		self.sess = sess
		self.data = data
		self.label = label
		self.embed = embedding
		self.train = train
		self.vocab_size = vocabulary_size

		self.train_x = self.data[:int(0.9*len(self.data))]
		self.train_y = self.label[:int(0.9*len(self.label))]

		self.test_x = self.data[int(0.9*len(self.data))+1:]
		self.test_y = self.label[int(0.9*len(self.label))+1:]

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

	def weight_variables(self, shape, name):
		"""
		Weight initialization with zero mean gaussian
		"""
		initial = tf.truncated_normal(shape, stddev=0.1, name="name")
		return tf.Variable(initial)

	def bias_variables(self, shape, name):
		initial = tf.constant(0.1, shape= shape, name="name")
		return tf.Variable(initial)

	def conv(self, input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):	
		#From https://github.com/ethereon/caffe-tensorflow
		c_i = input.get_shape()[-1]
		convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

		conv = convolve(input, kernel)
		return  conv

	def max_pool(self, x, k_h, k_w, s_h, s_w, padding):
		return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding= padding)

	def lrn(self, x, radius, alpha, beta, bias):
		return tf.nn.local_response_normalization( x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

	def FB_CNN_Entity(self, x):

		"""
		Given a cluster of entity embeddings computes a convolutional representation
		Returns a 1D tensor after passing through the convolution operators
		"""

		conv1_W = self.weight_variables([11, 1, 1, 1], name="conv1_W_e")
		conv1_b = self.bias_variables([1], name="conv1_b_e")
		conv1_in = tf.nn.conv2d(x, conv1_W, [1,3,1,1], padding="VALID")

		conv1 = tf.nn.relu(tf.nn.bias_add(conv1_in, conv1_b))
		pool1 = tf.nn.max_pool(conv1, [1,9,1,1], [1,9,1,1], padding="VALID")

		conv2_W = self.weight_variables([7, 1, 1, 1], name="conv2_W_e")
		conv2_b = self.bias_variables([1], name="conv2_b_e")
		conv2_in = tf.nn.conv2d(pool1, conv2_W, [1,2,1,1], padding="VALID")

		conv2 = tf.nn.relu(tf.nn.bias_add(conv2_in, conv2_b))
		pool2 = tf.nn.max_pool(conv2, [1,7,1,1], [1,7,1,1], padding="VALID")

		output = tf.reshape(pool2, [-1, 50])
		return output


	def FB_CNN_Relation(self, x):

		"""
		Given a cluster of relation embeddings computes a convolutional representation
		Returns a 1D tensor after passing through the convolution operators
		"""

		conv1_W = self.weight_variables([5, 1, 1, 1], name="conv1_W_r")
		conv1_b = self.bias_variables([1], name="conv1_b_r")
		conv1_in = tf.nn.conv2d(x, conv1_W, [1,2,1,1], padding="VALID")

		conv1 = tf.nn.relu(tf.nn.bias_add(conv1_in, conv1_b))
		pool1 = tf.nn.max_pool(conv1, [1,4,1,1], [1,4,1,1], padding="VALID")

		conv2_W = self.weight_variables([3, 1, 1, 1], name="conv2_W_r")
		conv2_b = self.bias_variables([1], name="conv2_b_r")
		conv2_in = tf.nn.conv2d(pool1, conv2_W, [1,2,1,1], padding="VALID")

		conv2 = tf.nn.relu(tf.nn.bias_add(conv2_in, conv2_b))
		pool2 = tf.nn.max_pool(conv2, [1,2,1,1], [1,2,1,1], padding="VALID")

		output = tf.reshape(pool2, [-1, 50])
		return output

	def LSTM_Entity(self, input_seq, weight, bias):

		"""
		LSTM module for input sentence sequence used for entity retrieval
		Processes sentences using dynamic length of sequences
		Returns the mean of all the hidden layers passed through a non-linearity ($C_E$)
		"""

		with tf.variable_scope("Entity_LSTM"):
			X = tf.nn.embedding_lookup(self.embed_init, input_seq)

			cell = tf.contrib.rnn.LSTMCell(self.n_hidden1, state_is_tuple = True, initializer= self.initializer())
			output, state = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32, sequence_length = self.length(X))
			output_all = tf.reduce_mean(output,1)
			output_all = tf.nn.relu(tf.add(tf.matmul(output_all, weight),bias))
			return output_all

	def LSTM_Relation(self, input_seq, weight, bias):

		"""
		LSTM module for input sentence sequence used for relation retrieval
		Processes sentences using dynamic length of sequences
		Returns the mean of all the hidden layers passed through a non-linearity ($C_R$)
		"""

		with tf.variable_scope("Relation_LSTM"):
			X = tf.nn.embedding_lookup(self.embed_init, input_seq)

			cell = tf.contrib.rnn.LSTMCell(self.n_hidden1, state_is_tuple = True, initializer= self.initializer())
			output, state = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32, sequence_length = self.length(X))
			output_all = tf.reduce_mean(output,1)
			output_all = tf.nn.relu(tf.add(tf.matmul(output_all, weight),bias))
			return output_all

	def KG_retrieval(self):

		"""
		Receives the entire entity and relation embedding space
		Returns the clustered entity and relation embedding space
		"""

		ent, rel = graph_reader.reader(self.entity_path, self.relation_path, self.entity_dim)

		while(len(ent)%20 !=0):
			ent.append([0]*self.entity_dim)

		while(len(rel)%20 != 0):
			rel.append([0]*self.relation_dim)

		self.entity_batch = len(ent)//self.n_clusters
		self.relation_batch = len(rel)//self.n_clusters

		for i in range(20):
			x = tf.reshape(ent[i*self.entity_batch: (i+1)*self.entity_batch], [1, self.entity_batch, self.entity_dim, 1])
			y = tf.reshape(rel[i*self.relation_batch: (i+1)*self.relation_batch], [1, self.relation_batch, self.relation_dim, 1])
			e = self.FB_CNN_Entity(x)
			r = self.FB_CNN_Relation(y)

			if i==0:
				ent_all = e 
				rel_all = r
			else:
				ent_all = tf.concat([ent_all, e], 0)
				rel_all = tf.concat([rel_all, r], 0)

		return ent_all, rel_all

	def attention(self, input_seq):

		"""
		Receives the input sentence batch
		Forms attention based on the sentences on the clustered embedding space
		Returns the resultant entity & relation for the batch
		"""

		ent, rel = self.KG_retrieval()

		with tf.variable_scope("entity_attn"):
			W_e = self.weight_variables([self.n_hidden1, self.n_hidden2], name="ent_w")
			B_e = self.bias_variables([self.n_hidden2], name="ent_b")
			y_all_E = self.LSTM_Entity(input_seq, W_e, B_e)

			self.attn_E = tf.transpose(tf.matmul(ent, tf.transpose(y_all_E)))
			self.attn_E = tf.nn.softmax(self.attn_E)

		with tf.variable_scope("relation_attn"):
			W_r = self.weight_variables([self.n_hidden1, self.n_hidden2], name="ent_r")
			B_r = self.bias_variables([self.n_hidden2], name="ent_r")
			y_all_R = self.LSTM_Relation(input_seq, W_r, B_r)

			self.attn_R = tf.transpose(tf.matmul(rel, tf.transpose(y_all_R)))
			self.attn_R = tf.nn.softmax(self.attn_R)

		entity = tf.matmul(self.attn_E, ent)
		relation = tf.matmul(self.attn_R, rel)
		return entity, relation

	def classify(self, input_seq):

		"""
		Receives the input sentence batch, retrieves the resultant entity & relation vector
		Concatenates the vectors & passes through a linear classifier to form the output
		"""

		self.entity, self.relation = self.attention(input_seq)
		self.input = tf.concat([self.entity, self.relation], 1)

		w_co = self.weight_variables([2*self.entity_dim, self.n_classes], name="cnn_w")
		b_co = self.bias_variables([self.n_classes], name="cnn_b")
		self.y_out = tf.nn.relu(tf.add(tf.matmul(self.input, w_co), b_co), name="output")

	def run(self):

		"""
		Driver function to train the entire model for SNLI
		"""
		
		self.W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.n_input]), trainable=False, name="W")
		self.embedding_placeholder = tf.placeholder("float", [None, None], name="embeds")
		self.embed_init = self.W.assign(self.embedding_placeholder)
		x = tf.placeholder("int64", [None, self.n_steps], name="data")
		y = tf.placeholder("int64", [None, self.n_classes], name="label")
		
		self.total_batches = len(self.train_x)//self.batch_size
		self.classify(x)

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= self.y_out))
		optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
		correct_prediction = tf.equal(tf.argmax(self.y_out,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('Test Accuracy', accuracy)

		merged = tf.summary.merge_all()

		if self.train:

			try:
				tf.global_variables_initializer().run()
			except:
				tf.initialize_all_variables().run()
			
			writer = tf.summary.FileWriter("kgcnn/logs/", graph=tf.get_default_graph())
			train_writer = tf.summary.FileWriter('kgcnn/logs/', self.sess.graph)

			for epoch in range(self.training_epochs):
				epoch_loss = 0
				for batch in range(self.total_batches):
					batch_x = self.train_x[batch*self.batch_size: (batch+1)*self.batch_size]
					batch_y = self.train_y[batch*self.batch_size: (batch+1)*self.batch_size]
					_, c, acc = self.sess.run([optimizer, loss, accuracy],feed_dict={x: batch_x, 
																		y : batch_y, 
																	  self.embedding_placeholder: self.embed})	
					epoch_loss += c
					logger.info(" Batch: {batch}: Loss = {loss}, Mini-batch accuracy: {acc}".format(batch=batch, loss=c,
																									acc= acc*100.0))

				summary, acc = self.sess.run([merged, accuracy], feed_dict={x: self.test_x, y: self.test_y, 
																			self.embedding_placeholder: self.embed})

				logger.info(" Epoch {epoch}: {avg_loss} Test Accuracy: {acc}".format(epoch=epoch, 
																avg_loss=1.0*epoch_loss/self.total_batches,
																acc = float(acc)*100))
				train_writer.add_summary(summary, epoch)
			
				saver = tf.train.Saver()				
				saver.save(self.sess, 'kgcnn/KG-CNN-model')

