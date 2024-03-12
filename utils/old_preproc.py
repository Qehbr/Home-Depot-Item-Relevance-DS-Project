import re

from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_html_tags_old(text):
    """
    Remove HTML tags from text
    :param text: text to delete HTML tags
    :return: filtered text
    """
    clean_text = re.sub(r'<[A-z]*', ' ', text)
    clean_text = re.sub(r'lt/>', ' ', clean_text)
    clean_text = re.sub(r'A href=".*"', ' ', clean_text)
    clean_text = re.sub(r'/A', ' ', clean_text)
    clean_text = re.sub(r'gt/>', ' ', clean_text)
    clean_text = re.sub(r'br>', ' ', clean_text)
    return re.sub(r'\s+', ' ', clean_text)



def remove_stopwords(text):
    """
    Removes stop words from text
    :param text: text to remove stop words
    :return: filtered text
    """
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


def remove_spaces(text):
    """
    Remove extra spaces from text
    :param text: text to remove spaces
    :return: filtered text
    """
    return re.sub(r'\s+', ' ', text)


def label_min_max_scaling(relevance):
    """
    Applies min max scaling to label
    :param relevance: to scale
    :return: scaled label
    """
    return (relevance - 1) / (3 - 1)


def inverse_label_min_max_scaling(relevance):
    """
    Inverses min max scaling of label
    :param relevance: to unscale
    :return: unscaled label
    """
    return relevance * (3 - 1) + 1








