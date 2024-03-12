import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Dataset class for model
    """

    def __init__(self, search_terms, product_descriptions, relevances, char_to_int, max_search, max_desc):
        """
        Constructor for CharDataset class
        :param search_terms: Series of search terms
        :param product_descriptions: Series of product descriptions
        :param relevances: Series of relevance (labels)
        :param char_to_int: Char to in mapping
        :param max_search: Max length of search term (used in padding)
        :param max_desc: Max length of description (used in padding)
        """
        self.search_terms = search_terms
        self.product_descriptions = product_descriptions
        self.relevances = relevances
        self.char_to_int = char_to_int
        self.max_search = max_search
        self.max_desc = max_desc

    def __len__(self):
        """
        Returns length of dataset
        :return: length of dataset
        """
        return len(self.search_terms)

    def __getitem__(self, idx):
        """
        Returns item from dataset
        :param idx: index
        :return: item from dataset
        """

        # convert sequences of characters to ints
        search_term = [self.char_to_int[char] for char in self.search_terms[idx] if char in self.char_to_int]
        product_description = [self.char_to_int[char] for char in self.product_descriptions[idx] if
                               char in self.char_to_int]

        # pad to make the same size
        search_term = self.pad(search_term, self.max_search)
        product_description = self.pad(product_description, self.max_desc)

        relevance = self.relevances[idx]
        return (torch.tensor(search_term, dtype=torch.long), torch.tensor(product_description, dtype=torch.long),
                torch.tensor(relevance, dtype=torch.float))

    def pad(self, characters, max_len):
        """
        Pads sequences
        :param characters: List of mapped to int characters
        :param max_len: Max length of padding
        :return: Padded sequence
        """
        if len(characters) < max_len:
            return characters + [0] * (max_len - len(characters))
        return characters[:max_len]
