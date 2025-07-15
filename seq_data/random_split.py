from random import sample
from Bio import SeqIO
import numpy as np
import pandas as pd

count = 0
Pos_SeqPair = {}
for record in SeqIO.parse("neg_train_ds2.fasta", "fasta"):
    UniProtID = str(record.id)
    ProtSeq = str(record.seq)
    Pos_SeqPair[UniProtID] = ProtSeq
    count += 1

Pos_list = []
Pos_list = list(Pos_SeqPair.keys())
print('ori:' + str(count))
test_ID_list = sample(Pos_list, int(146))
print('ext:' + str(len(test_ID_list)))
test_seq = open('neg_valid2_ds2.fasta', 'w')
for id in test_ID_list:
    test_seq.write(">" + str(id) + '\n')
    test_seq.write(Pos_SeqPair[id] + '\n')
    del Pos_SeqPair[id]
print(len(Pos_SeqPair))

final_len = len(Pos_SeqPair)
train_seq = open('neg_retrain2_ds2.fasta', 'w')
for key in Pos_SeqPair:
    train_seq.write(">" + str(key) + '\n')
    train_seq.write(Pos_SeqPair[key] + '\n')
