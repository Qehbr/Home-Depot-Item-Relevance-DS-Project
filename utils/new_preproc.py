import re

import nltk
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def remove_html_tags(text):
    """
    Remove HTML tags from text
    :param text: text to delete HTML tags
    :return: filtered text
    """
    # Remove standard HTML tags
    clean_text = re.sub(r'<[A-z]*', ' ', text)
    clean_text = re.sub(r'lt/>', ' ', clean_text)
    clean_text = re.sub(r'A href=".*"', ' ', clean_text)
    clean_text = re.sub(r'/A', ' ', clean_text)
    clean_text = re.sub(r'gt/>', ' ', clean_text)
    clean_text = re.sub(r'br>', ' ', clean_text)
    return clean_text


def remove_characters(s):
    """
    Remove characters from text
    :param text: text to delete characters
    :return: filtered text
    """
    # remove html tahs
    cleaned = remove_html_tags(s)

    # transform abbreviations
    cleaned = re.sub(r'in\.', r' inch ', cleaned)
    cleaned = re.sub(r'[Oo]z\.', r' ounce ', cleaned)
    cleaned = re.sub(r'[Ll]b\.', r' pound ', cleaned)
    cleaned = re.sub(r'[Ss]q\.', r' squared ', cleaned)
    cleaned = re.sub(r'[Ff]t\.', r' feet ', cleaned)
    cleaned = re.sub(r'[Gg]al\.', r' gallon ', cleaned)
    cleaned = re.sub(r'[Cc]u\.', r' cubic ', cleaned)
    cleaned = re.sub(r'&', r' and ', cleaned)
    cleaned = re.sub(r'%', r' percent ', cleaned)

    # additional cleaning
    cleaned = re.sub(r'"', r' ', cleaned)
    cleaned = re.sub(r"'", r' ', cleaned)
    cleaned = re.sub(r'-', r' ', cleaned)
    cleaned = re.sub(r'\+', r' ', cleaned)
    cleaned = re.sub(r':', r' ', cleaned)
    cleaned = re.sub(r'([a-z])([A-Z0-9])', r'\1 \2', cleaned)
    cleaned = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', cleaned)

    cleaned = re.sub(r'\. ([A-Z])', r' \1', cleaned)
    cleaned = re.sub(r'\.([A-Z])', r' \1', cleaned)
    cleaned = re.sub(r'\. ', ' ', cleaned)
    cleaned = re.sub(r',', ' ', cleaned)
    cleaned = re.sub(r';', ' ', cleaned)
    cleaned = re.sub(r'\((.*)\)', r'\1 ', cleaned)

    cleaned = re.sub(r'\(', r' ', cleaned)
    cleaned = re.sub(r'\)', r' ', cleaned)

    # extrac spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.lower()
    return cleaned


def preprocess_text(text, drop_stopwords):
    """
    Preprocess all text
    :param text: text to preprocess
    :param drop_stopwords: boolean to drop stopwords
    :return: preprocess tokens
    """
    text = remove_characters(text)

    # tokenization
    tokens = word_tokenize(text)

    # removing stop words
    if drop_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    # lemmatization
    tokens = [lemmatize_with_pos(word) for word in tokens]

    return tokens


def get_wordnet_pos(treebank_tag):
    """
    Converts treebank tags to wordnet tags
    :param treebank_tag: tag of treebank
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def lemmatize_with_pos(word):
    """
    Lemmatizes words based on their POS tags
    :param word: word to lemmatize
    """
    if word:
        pos = get_wordnet_pos(nltk.pos_tag([word])[0][1])
        return lemmatizer.lemmatize(word, pos=pos) if pos else word
    return word


def min_max_scaling(relevance):
    return 1 - (relevance - 1) / (3 - 1)


def inverse_min_max_scaling(relevance):
    return (1 - relevance) * (3 - 1) + 1