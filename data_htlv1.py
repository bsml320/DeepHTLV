#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
#from util import seq_matrix


# In[ ]:


def seq_matrix(seq_list, dim):
	tensor = np.zeros((len(seq_list),dim,4))
	y = []
	for i in range(len(seq_list)):
		seq = seq_list[i]
		j = 0
		for s in seq:
			if s == 'A' or s == 'a':
				tensor[i][j] = [1,0,0,0]
			if s == 'T' or s == 't':
				tensor[i][j] = [0,1,0,0]
			if s == 'C' or s == 'c':
				tensor[i][j] = [0,0,1,0]
			if s == 'G' or s == 'g':
				tensor[i][j] = [0,0,0,1]
			if s == 'N':
				tensor[i][j] = [0,0,0,0]
			j += 1
	
	return tensor


# In[ ]:


def bed_to_fasta():
	beds = ['data/VIS_pos_final.bed','data/VIS_neg_final.bed']
	for bed in beds:
		os.system("bedtools getfasta -fi ../ref/hg19.fa -bed " + bed + " -fo " + bed + ".fasta")


# In[ ]:


def fasta_to_matrix():
	seq_name = ['data/VIS_pos_final.bed' , 'data/VIS_neg_final.bed']
	
	print(seq_name)
	dim = 1000
	print('seq')
	### Seq ###

	for name in seq_name:
		if 'pos' in name:
			print(name)
			y = []
			seq = []
			positive_seq_file = open(name +'.fasta')
			lines = positive_seq_file.readlines()
			positive_seq_file.close()
			for line in lines:
				line = line.strip()
				if line[0] == '>':
					y.append(1)
				else:
					seq.append(line)

			X1 = seq_matrix(seq,dim)
			print('pos_ending!')
			np.save(name.split('.')[0], X1)

		if 'neg' in name:
			print(name)
			y = []
			seq = []
			negative_seq_file = open(name +'.fasta')
			lines = negative_seq_file.readlines()
			negative_seq_file.close()
			for line in lines:
				line = line.strip()
				if line[0] == '>':
					y.append(0)
				else:
					seq.append(line)

			X0 = seq_matrix(seq,dim)
			print('neg_ending!')
			np.save(name.split('.')[0], X0)

	X = np.concatenate([X1,X0])
	y = np.concatenate([np.ones(len(X1)), np.zeros(len(X0))])

	np.save('data/x_VISDB_fulldata.npy', X)
	np.save('data/y_VISDB_fulldata.npy', y)


# In[ ]:


if __name__ == '__main__':
	bed_to_fasta()
	fasta_to_matrix()


# In[ ]:




