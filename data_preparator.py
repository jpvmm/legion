import pandas as pd
from torchnlp import text_encoders
from torchnlp.utils import pad_tensor
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

LABELS = ['cma','cmq', 'cms', 'csa', 'csq',	'css', 'ot']

def load_dataset(path):
    df = pd.read_csv(path, dtype=str)

    tst = df[df.Label != '0']

    utterance = []
    allo = []
    tmpList = []
    labels = []
    for index, row in tst.iterrows():
        allo.append(row['Conversa'])
        labels.append(row['Label'])
        if row['Unnamed: 0'] == '0':
            if len(tmpList) != 0:
                utterance.append(tmpList)
                tmpList = []

        tmpList.append({'Conversa': row['Conversa'], 'Label': row['Label']})

    return allo, labels

def encode_text(lines):
    encoded = []

    #Cria o encoder
    encoder = text_encoders.StaticTokenizerEncoder(lines, tokenize=lambda s: s.split())

    vocab_size = encoder.vocab_size

    #Cria iterator para o encoder
    encoded = [encoder.encode(linha) for linha in lines]
    
    x_lengths = [len(sentence) for sentence in encoded]

    #Faz o padding de cada linha nos dialogos para um tamanho maximo de 90
    padded = [pad_tensor(encoded[cvs], length=90) for cvs in range(len(encoded))]

    return torch.stack(padded), vocab_size, x_lengths

def hot_encoder(labels):
#     le = LabelEncoder()

#     inte = le.fit_transform(labels)
    
#     inte = inte.reshape(len(labels),1)
    
    enc = OneHotEncoder(categories='auto')
    hot = enc.fit_transform(np.array(labels).reshape(len(labels),1))
    
    return hot    

def hot_decoder(pred):
       
    le = LabelEncoder()
    inte = le.fit_transform(LABELS)
    
    inteiro = np.argmax(pred)
    
    return le.inverse_transform([inteiro])    


def prepare_dataset(path, pred = False):
    lines, labels = load_dataset(path)

    padded, vocab_size, x_lengths = encode_text(lines)
    
    labels = hot_encoder(labels)
    
    return padded, labels.toarray(), vocab_size, x_lengths