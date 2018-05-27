import os
import sys


def reader(entity_file, relation_file, dim):

	"""
	Function returns entity and relation embeddings of Freebase formed using DKRL 
	"""

	entity = open(entity_file, "r")
	relation = open(relation_file, "r")

	ent_num = sum(1 for line in open(entity_file))
	rel_num = sum(1 for line in open(relation_file))

	print 'Number of entities %d' %ent_num
	print 'Number of relations %d' %rel_num

	ent = [ent[:] for ent in [[0] * dim] * ent_num]
	i = 0

	for line in entity:
		vec = line.split("\t")
		vec = vec[:-1]
		vec = [float(x) for x in vec]
		ent[i] = vec
		i += 1

	rel = [rel[:] for rel in [[0] * dim] * rel_num]
	i = 0

	for line in relation:
		vec = line.split("\t")
		vec = vec[:-1]
		vec = [float(x) for x in vec]
		rel[i] = vec
		i += 1

	return ent, rel

