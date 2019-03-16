import os
import sys
import time
import numpy as np
from random import shuffle
from config import *

def save_vocab(data):
	"""
	Saves the dataset vocabulary in a separate txt file
	"""
	print 'Vocabulary set: %d' %(len(data))
	f = open('vocabulary.txt',"w")
	for i in range(len(data)):
		f.write(data[i]+'\n')
	f.close()


def build_vocab(filepath):
	"""
	Browses the entire dataset to form the vocabulary array
	"""
	data = []

	for path, subdirs, files in os.walk(filepath):
		for filename in files:

			content = open(path+"/"+str(filename)).read().lower().replace("\n"," ").split(' ')
			data = data+content
			print len(data)

	data = list(set(data))
	print('Saving Vocabulary...')
	save_vocab(data)
	print('Vocabulary saved.')

def get_vocab():
	"""
	Retrieves vocabulary from the saved txt file
	"""
	vocab = []
	f = open('vocabulary.txt','r')
	for line in f:
		vocab.append(line.rstrip())
	f.close()
	return vocab

def get_word_vectors():
	"""
	This function is used only while using pre-trained Glove vectors
	Receives all the vectors useful in our dataset
	"""
	vocab = get_vocab()
	word2vec = {}
	emp = [0.1]*300
	f_vec = open('word_vectors.txt',"w")
	start = time.time()
	print('Reading word vectors....')
	f = open('../Dataset/glove.840B.300d.txt')

	for line in f:
		words = line.split()
		word = words[0]
		if word in vocab:
			vector = [float(val) for val in words[1:]]
			word2vec[word] = np.asarray(vector)
			vocab.remove(word)
			f_vec.write(line)

	f.close()
	end = time.time()
	print('Completed reading.')
	print 'Time taken %lf'%(end-start)
	
	start = time.time()
	print('Reading GloVe vectors...')
	f = open('../../GloVe/vectors.txt')
	word2vec_local = {}
	for line in f:
		words = line.split()
		word = words[0]
		vector = [float(val) for val in words[1:]]
		word2vec_local[word] = np.asarray(vector)
	f.close()
	end = time.time()
	print('Completed reading.')
	print 'Time taken %lf'%(end-start)

	for word in vocab:
		if word in word2vec_local.keys():
			word2vec[word] = word2vec_local[word]
		else:
			res = [words for words in word2vec_local.keys() if word in words]
			if len(res)!=0:
				word2vec[word] = word2vec_local[res[0]]
			else:
				word2vec[word] = emp
		s = word + ' ' + ' '.join(map(str, word2vec[word]))
		f_vec.write(s+os.linesep)

	f_vec.close()

def get_data(filepath, mode, lookup):
	"""
	Retrieves raw data from the dataset
	"""
	f_src = open(filepath+'src-'+mode+'.txt')
	f_targ = open(filepath+'targ-'+mode+'.txt')
	f_label = open(filepath+'label-'+mode+'.txt')

	data_src = []
	data_targ = []
	label = []

	for line in f_src:
		content = line.rstrip().lower().split(' ')
		sentence = []
		for i in range(len(content)):
			sentence.append(int(lookup[content[i]]))
		for j in range(i, 85):
			sentence.append(0);
		data_src.append(sentence)

	for line in f_targ:
		content = line.rstrip().lower().split(' ')
		sentence = []
		for i in range(len(content)):
			sentence.append(int(lookup[content[i]]))
		for j in range(i, 85):
			sentence.append(0);
		data_targ.append(sentence)

	for line in f_label:
		line = line[:-1]
		if(str(line) == "neutral"):
			label.append(0)
		if(str(line) == "entailment"):
			label.append(1)
		if(str(line) == "contradiction"):
			label.append(2)
 
	return data_src, data_targ, label			

def get_embedding():
	"""
	Collects the embedding vectors from the saved utility word vectors txt file
	"""
	f_vec = open('word_vectors.txt')
	lookup = {}
	embedding = []
	embedding.append([0]*300)
	i = 1
	print 'Reading the embeddings.....'
	start = time.time()
	for line in f_vec:
		words = line.split()
		word = words[0]
		vector = [float(val) for val in words[1:]]
		
		lookup[word] = i
		embedding.append(vector)
		i =  i + 1
	f_vec.close()
	end = time.time()
	print 'Completed.'
	print 'Time taken {:g}'.format(end-start)
	return lookup, embedding

def get_shuffled_dataset(filepath, mode, lookup):
	"""
	Returns a shuffled dataset useful for training purposes
	"""
	data1_, data2_, label_ = get_data(filepath, mode, lookup)
	data1 = []
	data2 = []
	label = []
	idx = range(len(data1_))
	shuffle(idx)
	for i in idx:
		data1.append(data1_[i])
		data2.append(data2_[i])
		lb = [0]*3
		lb[label_[i]] = 1
		label.append(np.asarray(lb))
	return data1, data2, label

def get_data2vec(filepath):

	"""
	Converts dataset to word vectors
	"""

	lookup, embedding = get_embedding()
	train_src, train_targ, train_label = get_shuffled_dataset(filepath, 'train', lookup)
	test_src, test_targ, test_label = get_shuffled_dataset(filepath, 'test', lookup)
	val_src, val_targ, val_label = get_shuffled_dataset(filepath, 'dev', lookup)

	return train_src, train_targ, train_label, test_src, test_targ, test_label, val_src, val_targ, val_label, embedding

def setup():
	build_vocab(DATASET_PATH)
	get_word_vectors()

if __name__=='__main__':
	# setup()
	get_data2vec('snli/')