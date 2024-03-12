import torch
from torch import nn
import torch.nn.functional as F


class CharSiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        """
        CharSiameseLSTM model constructor
        :param vocab_size: total vocabulary size
        :param embedding_dim: size of embedding
        :param hidden_dim: size of hidden layer in LSTM
        """
        super(CharSiameseLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.global_avg_pool = lambda x: torch.mean(x, dim=1)
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.batch_norm = nn.BatchNorm1d(1)

    def forward_once(self, x):
        """
        Forward Siamese network
        :param x: input
        :return: output of siamese network
        """
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.global_avg_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, inp_seq1, inp_seq2):
        """
        Forward call
        :param inp_seq1: first input (search terms)
        :param inp_seq2: second input (item descriptions)
        :return: output of the network
        """
        # get outputs of siamese networks
        output1 = self.forward_once(inp_seq1)
        output2 = self.forward_once(inp_seq2)

        # calculate distances
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1, keepdim=True))
        euclidean_distance = F.relu(euclidean_distance)  # Ensure non-negative
        euclidean_distance = self.batch_norm(euclidean_distance)
        output = torch.sigmoid(euclidean_distance)
        return output
