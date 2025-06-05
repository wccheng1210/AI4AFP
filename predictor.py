import argparse
from tensorflow.keras.models import load_model
from PC6_encoding import PC_6
from doc2vec import Doc2Vec_encoding
from bert_prediction import seq_bert_prediction
import joblib
import numpy as np
import pandas as pd

def main(fasta_path, output_name):
    # encoding
    dat = PC_6(fasta_path, 50)
    data_PC6 = np.array(list(dat.values()))
    data_Doc2Vec = Doc2Vec_encoding(fasta_path ,model_path='Doc2Vec_model/AFP_doc2vec.model')

    # reshape
    data_PC6_flatten = data_PC6.reshape(data_PC6.shape[0],-1)

    # load 7 models & predict
    PC6_nn_model = load_model('ensemble_model/pc6/pc6_best_weights.h5')
    PC6_nn_labels_score = PC6_nn_model.predict(data_PC6).reshape(-1)

    PC6_svc_model = joblib.load('ensemble_model/pc6/pc6_features_svm.pkl')
    PC6_svc_labels_score = PC6_svc_model.predict(data_PC6_flatten)

    PC6_rf_model = joblib.load('ensemble_model/pc6/pc6_features_forest.pkl')
    PC6_rf_labels_score = PC6_rf_model.predict(data_PC6_flatten)

    Doc2vec_nn_model = load_model('ensemble_model/doc2vec/doc2vec_best_weights.h5')
    Doc2vec_nn_labels_score = Doc2vec_nn_model.predict(data_Doc2Vec).reshape(-1)

    Doc2vec_svc = joblib.load('ensemble_model/doc2vec/doc2vec_features_svm.pkl')
    Doc2vec_svc_labels_score = Doc2vec_svc.predict(data_Doc2Vec)

    Doc2vec_rf_model = joblib.load('ensemble_model/doc2vec/doc2vec_features_forest.pkl')
    Doc2vec_rf_labels_score = Doc2vec_rf_model.predict(data_Doc2Vec)
    
    bert_test_score = seq_bert_prediction(fasta_path, 'Rostlab/prot_bert_bfd', './ensemble_model/bert', MAX_LEN=50)
    
    in_size = len(Doc2vec_svc_labels_score)
    pc6_thres = 0.4984
    doc2vec_thres = 0.7526
    bert_thres = 0.9892

    in_ensembleX_list = []
    for i in range(in_size):
        score_list = []
        score_list.append(float(PC6_rf_labels_score[i]))
        score_list.append(float(PC6_svc_labels_score[i]))
        if(PC6_nn_labels_score[i] >= pc6_thres):
            score_list.append(float('1'))
        else:
            score_list.append(float('0'))

        score_list.append(float(Doc2vec_rf_labels_score[i]))
        score_list.append(float(Doc2vec_svc_labels_score[i]))
        if(Doc2vec_nn_labels_score[i] >= doc2vec_thres):
            score_list.append(float('1'))
        else:
            score_list.append(float('0'))
            
        
        if(bert_test_score[i] >= bert_thres):
            score_list.append(float('1'))
        else:
            score_list.append(float('0'))
        
        in_ensembleX_list.append(score_list)
    in_ensembleX = np.array(in_ensembleX_list)

    ensemble_model = load_model('./ensemble_model/ensemble_best_weights.h5')
    pred_score = ensemble_model.predict(in_ensembleX)
    
    # make dataframe
    #for s in pred_score:
    classifier = pred_score > 0.8563
    df = pd.DataFrame(pred_score)
    df.insert(0,'Peptide' ,dat.keys())
    df.insert(2,'Prediction results', classifier)
    df['Prediction results'] = df['Prediction results'].replace({True: 'Yes', False: 'No'})
    df = df.rename({0:'Score'}, axis=1)
        # output csv
    df.to_csv(output_csv_name)

#arg
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='AI4Surfaceome ensemble predictor')
    parser.add_argument('-f','--fasta_name',help='input fasta path',required=True)
    parser.add_argument('-o','--output_csv',help='output csv name',required=True)
    args = parser.parse_args()
    fasta_path = args.fasta_name
    output_csv_name =  args.output_csv
    main(fasta_path, output_csv_name)