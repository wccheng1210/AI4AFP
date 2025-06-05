import os
import numpy as np
from doc2vec import read_fasta_to_kmers, train_doc2vec

# load data
AFP_data = read_fasta_to_kmers('./seq_data/merge_AFP_seq_len50(filter3011).fasta')
# pretrain doc2vec model
pretrain_d2v = './Doc2Vec_model' 
if not os.path.exists(pretrain_d2v):
    os.makedirs(pretrain_d2v)
train_doc2vec(AFP_data,'./Doc2Vec_model/AFP_doc2vec.model')
