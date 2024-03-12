import torch
from torch import nn
import torch.nn.functional as F


class BartSiamese(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim1=256, hidden_dim2=64):
        """
        BartSiamese model constructor
        :param embedding_dim: size of embedding
        :param hidden_dim1: size of second fc input
        :param hidden_dim2: size of third fc input
        """
        super(BartSiamese, self).__init__()

        # we experimented with different dropouts
        # self.dropout01 = nn.Dropout(0.1)
        # self.dropout02 = nn.Dropout(0.2)
        # self.dropout03 = nn.Dropout(0.3)
        # self.dropout04 = nn.Dropout(0.4)
        # self.dropout05 = nn.Dropout(0.5)

        self.fc = nn.Linear(embedding_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 128)
        self.batch_norm = nn.BatchNorm1d(1)

    def forward_once(self, x):
        """
        Forward Siamese network
        :param x: input
        :return: output of siamese network
        """
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def forward(self, inp_seq1, inp_seq2):
        """
        Forward call
        :param inp_seq1: first input (search terms)
        :param inp_seq2: second input (item descriptions)
        :return: output of the network
        """
        output1 = self.forward_once(inp_seq1)
        output2 = self.forward_once(inp_seq2)

        # calculate distances
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1, keepdim=True))
        euclidean_distance = F.relu(euclidean_distance)  # Ensure non-negative
        euclidean_distance = self.batch_norm(euclidean_distance)
        output = torch.sigmoid(euclidean_distance)
        return output

    def get_outputs(self, inp_seq1, inp_seq2):
        """
        Get outputs of siamese network without calculating distance
        :param inp_seq1: first input (search terms)
        :param inp_seq2: second input (item descriptions)
        :return: outputs of siamese network for each input
        """
        output1 = self.forward_once(inp_seq1)
        output2 = self.forward_once(inp_seq2)
        return output1, output2
