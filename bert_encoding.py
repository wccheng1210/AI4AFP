# BERT encoding
import torch
import random
import numpy as np
from Bio import SeqIO
from transformers import BertTokenizer
import random
import os

# read fasta as dict
def read_fasta(fasta_fname):
        path = fasta_fname
        r = dict()
        for record in SeqIO.parse(path, 'fasta'):
            idtag = str(record.id)
            seq = str(record.seq)
            r[idtag] = seq
        return r

def get_bert_features_labels(pos_fasta, neg_fasta, model_path, MAX_LEN):
    
    torch.cuda.set_device(0)

    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

    positive = read_fasta(pos_fasta)
    negtive = read_fasta(neg_fasta)

    positive_sentences = []
    negtive_sentences = []

    for p in positive.values():
        tmp = ''
        cnt = 0
        for w in p:
            cnt+=1
            if cnt<MAX_LEN-1:
                tmp+=w+' '
        positive_sentences.append(tmp)

    for n in negtive.values():
        tmp = ''
        cnt = 0
        for w in n:
            cnt+=1
            if cnt<MAX_LEN-1:
                try:
                    tmp+=w+' '
                except:
                    pass
        negtive_sentences.append(tmp)

    sentences=positive_sentences+negtive_sentences
    #labels = [1]*len(positive_sentences)+[0]*len(negtive_sentences)
    labels = np.hstack((np.repeat(1, len(positive_sentences)),np.repeat(0, len(negtive_sentences))))
 
    # 將數據集分完詞後儲存到列表中
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # 輸入文本
                            add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
                            max_length = MAX_LEN,           # 填充 & 截斷長度
                            pad_to_max_length = True,
                            return_attention_mask = True,   # 返回 attn. masks.
                            return_tensors = 'pt',     # 返回 pytorch tensors 格式的數據
                    )
    
        # 將編碼後的文本加入到列表  
        input_ids.append(encoded_dict['input_ids'][0].tolist())
    
        # 將文本的 attention mask 也加入到 attention_masks 列表
        attention_masks.append(encoded_dict['attention_mask'][0].tolist())

    # 將列表轉換為 tensor
    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(labels)

    # 輸出第1行文本的原始和編碼後的信息
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
    print('Attention Masks:', attention_masks[0])

    print('Min sentence length: ', min([len(sen) for sen in input_ids]))
    print('Max sentence length: ', max([len(sen) for sen in input_ids]))
    bert_features = np.array(input_ids)
    
    return bert_features, labels