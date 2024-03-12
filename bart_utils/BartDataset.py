import torch
from torch.utils.data import Dataset


class BartDataset(Dataset):
    def __init__(self, search_terms, product_descriptions, relevances):
        """
        Constructor for BartDataset class
        :param search_terms: Series of search terms (already embedded)
        :param product_descriptions: Series of product descriptions (already embedded)
        :param relevances: Series of relevance (labels)
        """
        self.search_terms = search_terms
        self.product_descriptions = product_descriptions
        self.relevances = relevances

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
        search_term_tensor = self.search_terms[idx]
        product_description_tensor = self.product_descriptions[idx]
        relevance_tensor = torch.tensor(self.relevances[idx], dtype=torch.float)
        return search_term_tensor, product_description_tensor, relevance_tensor
