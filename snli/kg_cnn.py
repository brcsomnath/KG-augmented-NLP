"""
This file implements the architecture of Convolutional based fact retrieval from Knowledge Graphs
Dataset: SNLI
Knowledge Graph: WordNet
"""


import tensorflow as tf 
import numpy as np 
import sys
import os
import time
import graph_reader
import timeit
import logging

from tensorflow.contrib.rnn.python.ops import rnn, rnn_cell

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KG_SNLI_CNN(object):

	"""
	Initializes parameters of the model from the driver file
	Implements the convolutional based architecture using attention mechanism and LSTMs
	Trains the model over specified number of epochs
	"""

	def __init__(self, sess, train_src, train_targ, train_label, test_src, test_targ, test_label, val_src,
				val_targ, val_label, embedding, entity_path = '../../ProjE/results/entity.txt',
				relation_path = '../../ProjE/results/relation.txt',	n_input = 300, n_steps = 86, train=False,
				n_hidden1 = 200, n_hidden2 = 25, n_classes = 3,  entity_dim = 50, relation_dim = 50,
				learning_rate = 0.001, training_epochs = 20, batch_size = 128, vocabulary_size = 36991):

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
		self.entity_dim = entity_dim
		self.relation_dim = relation_dim

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
			output_all = tf.nn.relu(tf.add(tf.matmul(output_all, weight),bias))
			return output_all

	def WN_CNN_Entity(self, x):
		"""
		Given a cluster of relation/ entity embeddings computes a convolutional representation
		Returns a 1D tensor after passing through the convolution operators
		"""

		conv1_W = self.weight_variable([15, 1, 1, 1], name="conv1_W_e")
		conv1_b = self.bias_variable([1], name="conv1_b_e")
		conv1_in = tf.nn.conv2d(x, conv1_W, [1,4,1,1], padding="VALID")

		conv1 = tf.nn.relu(tf.nn.bias_add(conv1_in, conv1_b))
		pool1 = tf.nn.max_pool(conv1, [1,13,1,1], [1,13,1,1], padding="VALID")

		conv2_W = self.weight_variable([11, 1, 1, 1], name="conv2_W_e")
		conv2_b = self.bias_variable([1], name="conv2_b_e")
		conv2_in = tf.nn.conv2d(pool1, conv2_W, [1,2,1,1], padding="VALID")

		conv2 = tf.nn.relu(tf.nn.bias_add(conv2_in, conv2_b))
		pool2 = tf.nn.max_pool(conv2, [1,11,1,1], [1,11,1,1], padding="VALID")

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
			return output

	def KG_retrieval(self):
		"""
		Receives the entire entity and relation embedding space
		Returns the clustered entity and relation embedding space
		"""
		ent, rel = graph_reader.reader(self.entity_path, self.relation_path, self.entity_dim)

		while(len(ent)%18 !=0):
			ent.append([0]*self.entity_dim)

		while(len(rel)%18 != 0):
			rel.append([0]*self.relation_dim)

		self.entity_batch = len(ent)//18
		self.relation_batch = len(rel)//18

		for i in range(18):
			x = tf.reshape(ent[i*self.entity_batch: (i+1)*self.entity_batch], [1, self.entity_batch, self.entity_dim, 1])
			y = tf.reshape(rel[i*self.relation_batch: (i+1)*self.relation_batch], [1, self.relation_batch, self.relation_dim, 1])
			e = self.WN_CNN_Entity(x)
			r = tf.reshape(y, [-1, self.relation_dim])
			if i==0:
				ent_all = e 
				rel_all = r
			else:
				ent_all = tf.concat([ent_all, e], 0)
				rel_all = tf.concat([rel_all, r], 0)

		return ent_all, rel_all

	def attention(self, input_src, input_targ):
		"""
		Receives the input sentence batch
		Forms attention based on the sentences on the clustered embedding space
		Returns the resultant entity & relation for the batch
		"""
		ent, rel = self.KG_retrieval()

		with tf.variable_scope("Entity_attn_src"):
			W_e1 = self.weight_variable([self.n_hidden1, self.n_hidden2], name="ent_w1")
			B_e1 = self.bias_variable([self.n_hidden2], name="ent_b1")
			y_E1 = self.LSTM_Entity(input_src, W_e1, B_e1)

		with tf.variable_scope("Entity_attn_targ"):
			W_e2 = self.weight_variable([self.n_hidden1, self.n_hidden2], name="ent_w2")
			B_e2 = self.bias_variable([self.n_hidden2], name="ent_b2")
			y_E2 = self.LSTM_Entity(input_targ, W_e2, B_e2)

		y_all_E = tf.concat([y_E1, y_E2], 1)
		self.attn_E = tf.transpose(tf.matmul(ent, tf.transpose(y_all_E)))
		self.attn_E = tf.nn.softmax(self.attn_E)

		logger.debug("Entity Attention Shape: {shape}".format(shape=self.attn_E.shape))

		with tf.variable_scope("Relation_attn_src"):
			W_r1 = self.weight_variable([self.n_hidden1, self.n_hidden2],name="rel_w1")
			B_r1 = self.bias_variable([self.n_hidden2],name="rel_b1")
			y_R1 = self.LSTM_Relation(input_src, W_r1, B_r1)


		with tf.variable_scope("Relation_attn_targ"):
			W_r2 = self.weight_variable([self.n_hidden1, self.n_hidden2],name="rel_w2")
			B_r2 = self.bias_variable([self.n_hidden2],name="rel_b2")
			y_R2 = self.LSTM_Relation(input_targ, W_r2, B_r2)

		y_all_R = tf.concat([y_R1, y_R2], 1)
		self.attn_R = tf.transpose(tf.matmul(rel, tf.transpose(y_all_R)))
		self.attn_R = tf.nn.softmax(self.attn_R)

		entity = tf.matmul(self.attn_E, ent)
		relation = tf.matmul(self.attn_R, rel)
		return entity, relation

	def classify(self, input_src, input_targ):
		"""
		Receives the input sentence batch, retrieves the resultant entity & relation vector
		Concatenates the vectors & passes through a linear classifier to form the output
		"""
		self.entity, self.relation = self.attention(input_src, input_targ)

		self.inp = tf.concat([self.entity, self.relation],1, name="output")
		w_co = self.weight_variable([2*self.entity_dim, self.n_classes], name="cnn_w")
		b_co = self.bias_variable([self.n_classes], name="cnn_b")
		
		self.y_out = tf.nn.relu(tf.add(tf.matmul(self.inp, w_co), b_co))

	def run(self):
		"""
		Driver function to train the entire model for SNLI
		"""
		self.W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.n_input]), trainable=False, name="W")
		self.embedding_placeholder = tf.placeholder("float", [None, None], name="embeds")
		self.embed_init = self.W.assign(self.embedding_placeholder)

		x1 = tf.placeholder("int64", [None, self.n_steps], name="input_src")
		x2 = tf.placeholder("int64", [None, self.n_steps], name="input_targ")
		y = tf.placeholder("int64", [None, self.n_classes], name="label")

		self.total_batches = len(self.train_src)//self.batch_size
		self.classify(x1, x2)

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= self.y_out))
		tf.summary.scalar('Validation Loss', loss)
		optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
		correct_prediction = tf.equal(tf.argmax(self.y_out,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('Validation Accuracy', accuracy)

		merged = tf.summary.merge_all()

		if self.train:
			try:
				tf.global_variables_initializer().run()
			except:
				tf.initialize_all_variables().run()

			writer = tf.summary.FileWriter("model/logs/", graph=tf.get_default_graph())
			train_writer = tf.summary.FileWriter('model/logs/', self.sess.graph)
			
			for epoch in range(self.training_epochs):
				epoch_loss = 0
				for batch in range(self.total_batches):
					
					batch_x1 = self.train_src[batch*self.batch_size: (batch+1)*self.batch_size]
					batch_x2 = self.train_targ[batch*self.batch_size: (batch+1)*self.batch_size]
					batch_y = self.train_label[batch*self.batch_size: (batch+1)*self.batch_size]
					_, c, acc = self.sess.run([optimizer, loss, accuracy],feed_dict={x1: batch_x1, x2: batch_x2,
																					 y : batch_y,
																					 self.embedding_placeholder: self.embed})
					epoch_loss += c
					
					print '\r',
					sys.stdout.flush()
					time.sleep(0.3)
					print 'Batch %d: Loss = %lf, Mini-batch accuracy: %lf ' %(batch, c, 1.0*acc*100.0),
				
				saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)				
				saver.save(self.sess, 'model/SNLI-lstm')

				summary, acc = self.sess.run([merged, accuracy], feed_dict={x1: self.val_src, x2: 
																			self.val_targ, y : self.val_label,
																			self.embedding_placeholder: self.embed})
				train_writer.add_summary(summary, epoch)
				logger.info(" Epoch {epoch}: {avg_loss} Validation Accuracy: {acc}".format(epoch=epoch, 
																					avg_loss=1.0*epoch_loss/self.total_batches,
																					acc = float(acc)*100))
			