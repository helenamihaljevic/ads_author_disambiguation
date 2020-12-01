import math
import re

import nltk
import pandas as pd
import string
import unidecode

# ACR_PATTERN = '\\b(?:[A-Z]){3,}'
from nltk import SnowballStemmer, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob as tb

ACR_PATTERN = '\\b(?<![<\/])(?:[A-Z]){3,}\\b'


def get_item_or_filler(l, pos, filler=''):
    """
    Return item of list 'l' at position 'pos' if it exists, otherwise return the 'filler' element.
    Example:
        get_item_or_filler(['a', 'b', 'c'], 1)
        >>>'b'
        get_item_or_filler(['a', 'b', 'c'], 4)
        >>>''
        get_item_or_filler(['a', 'b', 'c'], 4, 'e')
        >>>'e'
    """
    if len(l) > pos:
        return l[pos]
    else:
        return filler


def apply_unidecode(s):
    # This function removes accents, umlauts, and things like ø and the Polish l
    return unidecode.unidecode(s)


def remove_punctuation(s):
    """Strip out punctuation to simplify names"""
    punct = string.punctuation
    # Allow - as punctuation to preserve double names
    punct = punct.replace('-', '')
    translator = str.maketrans('', '', punct)
    return s.translate(translator)


def normalize_name(surname, forename):
    # Assumes we will be taking names with first+last format
    return '.'.join(['-'.join(surname.split()), '-'.join(forename.split())])


def reduce_ai(ai):
    try:
        sur, fore = ai.split('.')
        return '.'.join([sur, fore[0]])
    except:
        return None


def clean_name_parts(text):
    return apply_unidecode(remove_punctuation(text.lower())).strip().lstrip().lower()


def ignore_initials(s):
    if s.endswith('.') or len(s) <= 1:  # Russian initials like 'Yu. or 'H-K.', or regular initial without '.'
        return ''
    else:
        return s


def extract_surname_and_forenames(name_string):
    if name_string is not None:
        name_split = name_string.lower().split(',')

        if len(name_split) > 3 or len(name_split) == 1:  # TODO: make this better, see e.g. Reed, G. W., Jr.
            surname, forenames = '', ''
        else:
            name_split = [word.strip() for word in name_split]
            surname = name_split[0]
            forenames = name_split[1]
    else:
        surname, forenames = '', ''

    return forenames, clean_name_parts(surname)


def extract_first_and_middle_name(forenames, separator=None, ignore_init=True):
    if forenames is not None:  # not required for ads data but kept to have the same version for all databases
        forenames = str(forenames).lower()
        words = forenames.replace('-', ' ').replace('.', ' ').split(
            separator)  # using 'replace' instead of re.split() for speed
        first_name = get_item_or_filler(words, 0)
        middle_name = get_item_or_filler(words, 1)
        if ignore_init is True:
            first_name = ignore_initials(first_name)
            middle_name = ignore_initials(middle_name)
    else:
        first_name, middle_name = '', ''

    return clean_name_parts(first_name), clean_name_parts(middle_name)


def extract_name_parts(name):
    forenames, surname = extract_surname_and_forenames(name)
    first, middle = extract_first_and_middle_name(forenames, ignore_init=False)
    middle_name_init = middle[0] if len(middle) > 0 else ''
    first = clean_name_parts(ignore_initials(first))
    middle = clean_name_parts(ignore_initials(middle))

    return first, middle, clean_name_parts(middle_name_init)


def extract_middle_name_initial(name):
    first, middle, middle_init = extract_name_parts(name)
    return middle_init


def extract_acronyms(s, p=ACR_PATTERN):
    p_compiled = re.compile(p)
    st = '' if pd.isnull(s) else s
    return set(re.findall(p_compiled, st))


stemmer = SnowballStemmer("english")


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def tfidf(word, blob, bloblist):
    def tf(word, blob):
        return blob.words.count(word) / len(blob.words)

    def idf(word, bloblist):
        return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

    return tf(word, blob) * idf(word, bloblist)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def preprocess_text(text):
    # Lower case and strip out HTML tags
    text = cleanhtml(text).lower()
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    # Filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    # Also filter out short words with len < 3
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token) and (len(token) > 2):
            filtered_tokens.append(token)
    # Stem words
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def create_corpus_from_docs_keywords(docs):
    """Takes df containing publications & creates textblobs with all abstracts + titles preprocessed"""
    corpus = []
    for abst, tit in zip(docs.abstract, docs.title):
        if pd.isnull(abst) and pd.isnull(tit):
            continue
        tit = '' if pd.isnull(tit) else tit
        abst = '' if pd.isnull(abst) else abst
        text = ' '.join([tit, abst])
        preproc_text = ' '.join(preprocess_text(text))
        corpus.append(tb(preproc_text))
    return corpus


def create_corpus_from_docs(docs):
    corpus = {}

    for i, (abst, tit) in enumerate(zip(docs.abstract, docs.title)):
        if pd.isnull(abst) and pd.isnull(tit):
            continue
        tit = '' if pd.isnull(tit) else tit
        abst = '' if pd.isnull(abst) else abst
        text = ' '.join([tit, abst])
        corpus[i] = ' '.join(preprocess_text(text))

    return corpus


def extract_top_ngrams(doc, corpus, ngram_range=(1, 3), lim=10):
    tit = '' if pd.isnull(doc['title']) else doc['title']
    abst = '' if pd.isnull(doc['abstract']) else doc['abstract']
    text = ' '.join([tit, abst])

    tokens = preprocess_text(text)
    bigrams = [' '.join(b) for b in ngrams(tokens, 2)]
    trigrams = [' '.join(b) for b in ngrams(tokens, 3)]

    myvocabulary = list(set(tokens)) + list(set(bigrams)) + list(set(trigrams))

    # Prevent TfidfVectorizer error when passing empty vocabulary
    if myvocabulary == []:
        return []

    else:
        tfidf = TfidfVectorizer(vocabulary=myvocabulary, stop_words='english', ngram_range=ngram_range)
        tfs = tfidf.fit_transform(corpus.values())

        feature_names = tfidf.get_feature_names()
        corpus_index = [n for n in corpus]
        rows, cols = tfs.nonzero()
        return [a[0] for a in
                sorted([(feature_names[col], corpus_index[row], tfs[row, col]) for (row, col) in zip(rows, cols)],
                       key=lambda a: a[2], reverse=True)[:lim]]


def extract_top_keywords(doc, corpus, n=5):
    """Takes a document from a general corpus and returns the top n words.
    Uses title and abstract from the document
    Applies TF-IDF to rank importance of words
    Needs a corpus as produced by create_corpus_from_docs"""
    tit = '' if pd.isnull(doc['title']) else doc['title']
    abst = '' if pd.isnull(doc['abstract']) else doc['abstract']
    text = ' '.join([tit, abst])
    preproc_text = ' '.join(preprocess_text(text))
    blob = tb(preproc_text)
    scores = {word: tfidf(word, blob, corpus) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(word, score, len(blob.words)) for (word, score) in sorted_words[:n]]