import pandas as pd
from torchnlp import text_encoders
from torchnlp.utils import pad_tensor

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
    gen = (i for i in lines)
    encoded.append(encoder.encode(gen.__next__()))

    #Faz o padding de cada linha nos dialogos para um tamanho maximo de 90
    padded = [pad_tensor(encoded[cvs], length=90) for cvs in range(len(encoded))]

    return padded, vocab_size

def prepare_dataset(path):
    lines, labels = load_dataset(path)

    padded, vocab_size = encode_text(lines)

    return padded, labels, vocab_size