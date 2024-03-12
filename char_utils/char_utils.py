def to_character_sequences(text):
    """
    Converts a text to a sequence of characters
    :param text: text to convert
    :return: sequence of characters
    """
    return list(text.lower())


def create_char_to_int_mapping(search_terms, product_descriptions):
    """
    Creates a char_utils to int mapping
    :param search_terms: search terms to use for mapping
    :param product_descriptions: descriptions to use for mapping
    :return: mapping from char_utils to int
    """
    all_chars = set(''.join([''.join(seq) for seq in search_terms + product_descriptions]))
    # starting index from 1 to reserve 0 for padding
    char_to_int = {char: i + 1 for i, char in enumerate(all_chars)}
    return char_to_int
