import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DAC(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers):
        super(DAC, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size+1, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, n_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size, 7)

    def forward(self, input, sequencias):
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)

        embbeded = self.embedding(input)

        gru_input = pack_padded_sequence(embbeded, sequencias.data.cpu().numpy())

        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)

        fc_output = self.fc(hidden[-1])

        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * 2,
                             batch_size, self.hidden_size)

        return self.create_variable(hidden)

    def create_variable(self, tensor):
        # Do cuda() before wrapping with variable
        if torch.cuda.is_available():
            return Variable(tensor.cuda())
        else:
            return Variable(tensor)