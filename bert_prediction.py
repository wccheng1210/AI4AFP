# BERT encoding
import os
import pandas as pd
import numpy as np
import random
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from transformers import get_linear_schedule_with_warmup

# read fasta as dict
def read_fasta(fasta_fname):
        path = fasta_fname
        r = dict()
        for record in SeqIO.parse(path, 'fasta'):
            idtag = str(record.id)
            seq = str(record.seq)
            r[idtag] = seq
        return r
        
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_bert_prediction(train_pos_fasta, train_neg_fasta, test_pos_fasta, test_neg_fasta, model_name, model_path, MAX_LEN):
    #torch.cuda.set_device(0)
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
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    positive = read_fasta(train_pos_fasta)
    negtive = read_fasta(train_neg_fasta)
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
    labels = np.hstack((np.repeat(1, len(positive_sentences)),np.repeat(0, len(negtive_sentences))))
    
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True, # add '[CLS]' 和 '[SEP]'
                            max_length = MAX_LEN,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt'
                    )
        input_ids.append(encoded_dict['input_ids'][0].tolist())
        attention_masks.append(encoded_dict['attention_mask'][0].tolist())
    print('Min sentence length: ', min([len(sen) for sen in input_ids]))
    print('Max sentence length: ', max([len(sen) for sen in input_ids]))
    
    # Divide up our training set to use 90% for training and 10% for validation
    tmp = labels.copy()
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, tmp, 
                                                                random_state=319, test_size=0.1)
    
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, tmp,
                                                random_state=319, test_size=0.1)
    len(train_inputs),len(validation_inputs),len(train_masks),len(validation_masks)
    
    # Converting to PyTorch Data Types
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    
    batch_size = 1
    
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    # Train Our Classification Model
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False
    )
    
    # Tell pytorch to run this model on the GPU.
    #model.cuda()
    
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    import gc
    
    del tmp
    del input_ids
    del labels
    del train_data
    del train_sampler
    del validation_data
    del validation_sampler
    del attention_masks
    del train_inputs
    del train_masks
    del train_labels
    del validation_inputs
    del validation_masks
    del validation_labels
    del params
    
    gc.collect()
    
    # Optimizer & Learning Rate Scheduler
    learning_rate = 1e-6
    optimizer = AdamW(model.parameters(),
                    lr = learning_rate,
                    eps = 1e-8
                    )
    
    # Number of training epochs
    epochs = 1
    
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    # Training Loop
    # Change "fp16_training" to True to support automatic mixed precision training (fp16)
    fp16_training = False
    
    if fp16_training:
        #!pip install accelerate==0.2.0
        from accelerate import Accelerator
        accelerator = Accelerator(fp16=True)
        device = accelerator.device
    
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    avg_loss_values = []
    avg_accuracy = []
    avg_valid_loss_values = []
    avg_valid_accuracy = []
    lr_scheduler = False
    
    if fp16_training:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # For each epoch...
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
    
        # Measure how long the training epoch takes.
        t0 = time.time()
        eval_loss, eval_accuracy = 0, 0
        
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
        
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            
            loss = outputs[0]
            total_loss += loss.item()
            loss_values.append(loss.item())
            
            if fp16_training:
                accelerator.backward(loss)
            else:
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
            if lr_scheduler:
                scheduler.step()
            # ========================================          
            #            calculate accuracy
            # ========================================
            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
            # ========================================
            # Progress update every 50 batches.
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.   Elapsed: {:}.   Loss: {:}.'.format(step, len(train_dataloader), elapsed, loss.item()))
        
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            
        avg_train_acc = eval_accuracy / len(train_dataloader)            
        
        # Store the loss value for plotting the learning curve.
        avg_loss_values.append(avg_train_loss)
        avg_accuracy.append(avg_train_acc)
        
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Average training accuracy: {0:.2f}".format(avg_train_acc))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        # ========================================
        #               Validation
        # ========================================
        print("Running Validation...")
        
        t0 = time.time()
        model.eval()
        
        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        valid_total_loss = 0
        
        # Evaluate data for one epoch    
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            logits = outputs[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
            
            # Track the number of batches
            nb_eval_steps += 1
            
            with torch.no_grad():        
                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            loss = outputs[0]
            valid_total_loss+=loss.item()
        
        avg_valid_loss_values.append(valid_total_loss/nb_eval_steps)
        avg_valid_accuracy.append(eval_accuracy/nb_eval_steps)
        
        # Report the final accuracy for this validation run.
        print("  Loss: {0:.2f}".format(valid_total_loss/nb_eval_steps))
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        output_dir = model_path
        
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Saving model to %s" % output_dir)
        file_name = 'ensemble_prot_bert_bfd_epoch'+str(epoch_i+1)+'_'+str(learning_rate)+'.pt'
        torch.save(model,os.path.join(output_dir, file_name))
    print("Training complete!")
    
    # Performance On Test Set
    # Data Preparation
    test_positive = read_fasta(test_pos_fasta)
    test_negtive = read_fasta(test_neg_fasta)
    
    test_positive_sentences = []
    test_negtive_sentences = []
    for p in test_positive.values():
        tmp = ''
        cnt = 0
        for w in p:
            cnt+=1
            if cnt<MAX_LEN-1:
                tmp+=w+' '
        test_positive_sentences.append(tmp)
    for n in test_negtive.values():
        tmp = ''
        cnt = 0
        for w in n:
            cnt+=1
            if cnt<MAX_LEN-1:
                try:
                    tmp+=w+' '
                except:
                    pass
        test_negtive_sentences.append(tmp)
    
    test_sentences=test_positive_sentences+test_negtive_sentences
    test_labels = [1]*len(test_positive_sentences)+[0]*len(test_negtive_sentences)
    
    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(len(test_sentences)))
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    test_input_ids = []
    
    # For every sentence...
    for sent in test_sentences:
        encoded_sent = tokenizer.encode(
                            sent,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )
        test_input_ids.append(encoded_sent)

    for index, value in enumerate(test_input_ids):
        if len(value)<MAX_LEN:
            test_input_ids[index] = value+[0]*(MAX_LEN-len(value))
    
    # Create attention masks
    test_attention_masks = []
    
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in test_input_ids:
        seq_mask = [float(i>0) for i in seq]
        test_attention_masks.append(seq_mask) 
    
    # Convert to tensors.
    prediction_inputs = torch.tensor(test_input_ids)
    prediction_masks = torch.tensor(test_attention_masks)
    prediction_labels = torch.tensor(test_labels)
    
    # Set the batch size.  
    batch_size = 1
    
    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    # Evaluate on Test Set
    # With the test set prepared, we can apply our fine-tuned model to generate predictions on the test set.
    model = torch.load(model_path + '/ensemble_prot_bert_bfd_epoch1_1e-06.pt', weights_only = False)
    
    # Prediction on test set
    # Put model in evaluation mode
    model.eval()
    
    # Tracking variables 
    predictions , true_labels = [], []
    
    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
        
        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions+=list(logits)
        true_labels.append(label_ids)
    
    print('DONE.')
    
    from sklearn.metrics import roc_curve
    
    pred_labels = []
    labels_score = []
    predictions = np.array(predictions)
    labels_score = torch.nn.Softmax(dim=1)(torch.tensor(predictions)).numpy()
    
    fpr, tpr, thresholds = roc_curve(test_labels, labels_score[:,1])
    
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    pred_labels = [1 if p else 0 for p in labels_score[:,1] > thresholds[ix]]
    
    return pred_labels, thresholds[ix]

def run_bert_prediction(test_pos_fasta, test_neg_fasta, model_name, model_path, MAX_LEN):
    #torch.cuda.set_device(0)
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
    
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    
    # Performance On Test Set
    # Data Preparation
    test_positive = read_fasta(test_pos_fasta)
    test_negtive = read_fasta(test_neg_fasta)
    
    test_positive_sentences = []
    test_negtive_sentences = []
    
    for p in test_positive.values():
        tmp = ''
        cnt = 0
        for w in p:
            cnt+=1
            if cnt<MAX_LEN-1:
                tmp+=w+' '
        test_positive_sentences.append(tmp)
    
    for n in test_negtive.values():
        tmp = ''
        cnt = 0
        for w in n:
            cnt+=1
            if cnt<MAX_LEN-1:
                try:
                    tmp+=w+' '
                except:
                    pass
        test_negtive_sentences.append(tmp)
    
    test_sentences=test_positive_sentences+test_negtive_sentences
    test_labels = [1]*len(test_positive_sentences)+[0]*len(test_negtive_sentences)
    
    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(len(test_sentences)))
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    test_input_ids = []
    
    # For every sentence...
    for sent in test_sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )
        test_input_ids.append(encoded_sent)
    
    for index, value in enumerate(test_input_ids):
        if len(value)<MAX_LEN:
            test_input_ids[index] = value+[0]*(MAX_LEN-len(value))
    
    # Create attention masks
    test_attention_masks = []
    
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in test_input_ids:
        seq_mask = [float(i>0) for i in seq]
        test_attention_masks.append(seq_mask) 
    
    # Convert to tensors.
    prediction_inputs = torch.tensor(test_input_ids)
    prediction_masks = torch.tensor(test_attention_masks)
    prediction_labels = torch.tensor(test_labels)
    
    # Set the batch size.  
    batch_size = 1
    
    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        
    # Evaluate on Test Set
    model = torch.load(model_path + '/ensemble_prot_bert_bfd_epoch1_1e-06.pt', weights_only = False)
    
    # Prediction on test set
    model.eval()
    
    # Tracking variables 
    predictions , true_labels = [], []
    
    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
        # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
        
        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
      
        # Store predictions and true labels
        predictions+=list(logits)
        true_labels.append(label_ids)
    
    print('DONE.')
    
    from sklearn.metrics import roc_curve
    
    pred_labels = []
    labels_score = []
    predictions = np.array(predictions)
    labels_score = torch.nn.Softmax(dim=1)(torch.tensor(predictions)).numpy()
    
    fpr, tpr, thresholds = roc_curve(test_labels, labels_score[:,1])
    
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    
    pred_labels = [1 if p else 0 for p in labels_score[:,1] > thresholds[ix]]
    
    return pred_labels, thresholds[ix]

def seq_bert_prediction(seq_fasta, thres, model_name, model_path, MAX_LEN):
    #torch.cuda.set_device(0)
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
    
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    
    pred_seq = read_fasta(seq_fasta)
    pred_seq_sentences = []
    
    for s in pred_seq.values():
        tmp = ''
        cnt = 0
        for w in s:
            cnt+=1
            if cnt<MAX_LEN-1:
                tmp+=w+' '
        pred_seq_sentences.append(tmp)
    
    # Report the number of sentences.
    print('Number of seqence sentences: {:,}\n'.format(len(pred_seq_sentences)))
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    pred_seq_input_ids = []
    
    for sent in pred_seq_sentences:
        encoded_sent = tokenizer.encode(
                            sent,
                            add_special_tokens = True
                       )
        pred_seq_input_ids.append(encoded_sent)
    
    for index, value in enumerate(pred_seq_input_ids):
        if len(value)<MAX_LEN:
            pred_seq_input_ids[index] = value+[0]*(MAX_LEN-len(value))
    
    # Create attention masks
    pred_seq_attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in pred_seq_input_ids:
        seq_mask = [float(i>0) for i in seq]
        pred_seq_attention_masks.append(seq_mask) 
    
    # Convert to tensors.
    prediction_inputs = torch.tensor(pred_seq_input_ids)
    prediction_masks = torch.tensor(pred_seq_attention_masks)
    
    # Set the batch size.  
    batch_size = 1
    
    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        
    # Evaluate on Test Set
    model = torch.load(model_path + '/ensemble_prot_bert_bfd_epoch1_1e-06.pt', weights_only = False)
    
    # Put model in evaluation mode
    model.eval()
    
    # Tracking variables 
    predictions , true_labels = [], []
    
    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
    
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask= batch
    
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
        
        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
    
        # Store predictions and true labels
        predictions+=list(logits)
    
    print('DONE.')
    
    pred_labels = []
    predictions = np.array(predictions)
    labels_score = torch.nn.Softmax(dim=1)(torch.tensor(predictions)).numpy()
    
    pred_labels = [1 if p else 0 for p in labels_score[:,1] > thres]
    
    return pred_labels