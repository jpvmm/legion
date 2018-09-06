import pandas as pd
import ast
import string
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import _pickle as cPickle

use_cuda = True

def load_data():
    with open('full_convs.pickle', 'rb') as f:
        full_convs = cPickle.load(f)

    # Normalizando para tudo minusculo
    for i in range(len(full_convs)):
        full_convs[i] = [i.lower() for i in full_convs[i]]

    return full_convs

def to_dataframe(full_convs):
    ''' Converte para dataframe '''

    df_list = [pd.DataFrame(full_convs[line]) for line in range(len(full_convs))]

    df = pd.concat(df_list)

    df['label'] = 0

    df.columns = ['Conversa', 'Label']

    return df

if __name__ == '__main__':
    full_convs = load_data()

    df = to_dataframe(full_convs)


