import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import prepare_dataset
from torch import nn

EMBEDDING_DIM = 300
HIDDEN_DIM = 200
BATCH_SIZE = 256
SEQ_LEN = 300

class LSTMClassifier(nn.Module):

	"""
	Initializes parameters of the model from the driver file
	Implements the baseline model using LSTM
	Trains the model over specified number of epochs
	"""

	def __init__(self, hidden_dim, embeddings, tagset_size, batch_size):

		"""
		Model parameter initialization
		"""
		super(LSTMClassifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.tagset_size = tagset_size

		self.word_embeddings = nn.Embedding(len(embeddings), len(embeddings[0]))
		self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(embeddings))
		self.word_embeddings.weight.requires_grad = False

		self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
		        autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

	def forward(self, sentence):

		"""
		Receives the input sentence batch
		Acts as a driver function for the LSTM
		Returns the prediction after passing the LSTM vectors through a 1-layer NN
		"""
		
		embeds = self.word_embeddings(sentence)
		lstm_out, self.hidden = self.lstm(embeds.view(SEQ_LEN, self.batch_size, -1), self.hidden)
		tag_space = self.hidden2tag(lstm_out.view(-1, self.hidden_dim))
		tag_space = tag_space.view(-1, SEQ_LEN, self.tagset_size)
		output = torch.mean(tag_space, 1)
		output = output.view(-1, self.tagset_size)
		output = F.log_softmax(output)
		#print output.size()
		return output
