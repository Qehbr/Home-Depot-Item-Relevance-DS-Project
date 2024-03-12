import torch


def pad_collate_fn(batch):
    """
    Pads the batch with the max length of search and descriptions in current batch
    :param batch: to pad
    :return: padded batch
    """
    search_terms, product_descriptions, relevances = zip(*batch)

    # find the maximum length in this batch for search terms and product descriptions
    max_len_search = max([len(x) for x in search_terms])
    max_len_desc = max([len(x) for x in product_descriptions])

    # pad each sequence to the max length
    search_terms_padded = [torch.nn.functional.pad(x, (0, 0, 0, max_len_search - len(x))) for x in search_terms]
    product_descriptions_padded = [torch.nn.functional.pad(x, (0, 0, 0, max_len_desc - len(x))) for x in
                                   product_descriptions]

    search_terms_padded = torch.stack(search_terms_padded)
    product_descriptions_padded = torch.stack(product_descriptions_padded)
    relevances = torch.tensor(relevances, dtype=torch.float)

    return search_terms_padded, product_descriptions_padded, relevances
