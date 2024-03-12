import torch


def get_bart_embeddings(text, bart_tokenizer, bart_model):
    """
    Get bart embeddings from text
    :param text: text to be embedded
    :param bart_tokenizer: tokenizer
    :param bart_model: model
    :return: embedding
    """
    token = bart_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = bart_model(**token.to(bart_model.device)).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()[0]
