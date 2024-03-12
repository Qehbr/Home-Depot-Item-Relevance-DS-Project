import numpy as np
import torch
from torch.utils.data import Dataset


class WordDataset(Dataset):
    def __init__(self, search_terms, product_descriptions, relevances, word2vec):
        """
        Constructor for WordDataset class
        :param search_terms: Series of search terms
        :param product_descriptions: Series of product descriptions
        :param relevances: Series of relevance (labels)
        :param word2vec: word2vec model
        """
        self.search_terms = search_terms
        self.product_descriptions = product_descriptions
        self.relevances = relevances
        self.word2vec_model = word2vec

    def __len__(self):
        """
        Returns length of dataset
        :return: length of dataset
        """
        return len(self.relevances)

    def __getitem__(self, idx):
        """
        Returns item from dataset
        :param idx: index
        :return: item from dataset
        """

        search_term = self.search_terms[idx]
        product_description = self.product_descriptions[idx]

        # convert to vectors
        search_term_vectors = [self.word2vec_model.wv[token] for token in search_term if
                               token in self.word2vec_model.wv]
        product_description_vectors = [self.word2vec_model.wv[token] for token in product_description if
                                       token in self.word2vec_model.wv]

        # convert lists of vectors to np arrays
        search_term_vectors_np = np.array(search_term_vectors, dtype=np.float32)
        product_description_vectors_np = np.array(product_description_vectors, dtype=np.float32)

        relevance_tensor = torch.tensor(self.relevances[idx], dtype=torch.float)
        search_term_tensor = torch.from_numpy(search_term_vectors_np)
        product_description_tensor = torch.from_numpy(product_description_vectors_np)
        return search_term_tensor, product_description_tensor, relevance_tensor
