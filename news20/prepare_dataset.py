import os 
import sys
import numpy as np
import tensorflow as tf
import clean_dataset
from random import shuffle
import time

def save_vocab(data):

	"""
	Saves the dataset vocabulary in a separate txt file
	"""

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
			content = open(path+"/"+str(filename)).read().split(' ')
			data = data+content
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
	f = open('Dataset/glove.840B.300d.txt')

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
	f = open('../GloVe/vectors.txt')
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
				print word
				word2vec[word] = emp
		s = word + ' ' + ' '.join(map(str, word2vec[word]))
		f_vec.write(s+os.linesep)

	f_vec.close()
	return word2vec


def get_dummy_word_vec():
	vocab = get_vocab()
	word2vec = {}
	dummy = [0.1]*300

	for i in range(len(vocab)):
		word2vec[vocab[i]] = dummy
	return word2vec


def get_data2vec():
	"""
	Converts dataset to word vectors
	"""
	data_, label = clean_dataset.retrieve_data_files()
	data = []
	lookup, embedding = get_embedding()

	start = time.time()
	print 'Converting dataset from words to vectors........'
	for i in range(len(data_)):
		sentence = []
		for j in range(300):
			if j<len(data_[i]):
				sentence.append(int(lookup[data_[i][j]]))
			else:
				sentence.append(0)
		data.append(sentence)
		end = time.time()
	print 'Completed.'
	print 'Time taken %lf'%(end-start)

	return data, label, embedding

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
	#embedding = np.asarray(embedding)
	return lookup, embedding



def get_shuffled_dataset():
	"""
	Returns a shuffled dataset useful for training purposes
	"""
	data_, label_, embedding= get_data2vec()
	data = []
	label = []
	idx = range(len(data_))
	shuffle(idx)
	for i in idx:
		data.append(data_[i])
		lb = [0]*20
		lb[label_[i]] = 1
		label.append(np.asarray(lb))
	return data, label, embedding


if __name__=='__main__':
	_, _, embed = get_shuffled_dataset()
