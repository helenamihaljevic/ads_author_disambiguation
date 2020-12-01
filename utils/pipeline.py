import pickle
import re
from os.path import join

# from similarity.jarowinkler import JaroWinkler
import editdistance
import numpy as np
import phonetics
from nltk import word_tokenize
from scipy.spatial.distance import cosine
from textdistance import jaro_winkler
from itertools import product

from config import PATH_DATA
from utils.helpers import MISSING_VAL_FILLER, is_missing
from utils.text_proc import extract_surname_and_forenames, normalize_name, clean_name_parts, reduce_ai

handle_nicknames = open(join(PATH_DATA, 'name_to_nickname.pickle'), 'rb')
NAME_TO_NICKNAME = pickle.load(handle_nicknames)

handle_name_vars = open(join(PATH_DATA, 'name_variations.pickle'), 'rb')
NAME_VARIATIONS = pickle.load(handle_name_vars)

handle_acronyms = open(join(PATH_DATA, 'acronym_to_idf.pickle'), 'rb')
ACRONYM_TO_IDF = pickle.load(handle_acronyms)

handle_arxiv_class = open(join(PATH_DATA, 'arxiv_class_to_idf.pickle'), 'rb')
ARXIV_CLASS_TO_IDF = pickle.load(handle_arxiv_class)

handle_keywords = open(join(PATH_DATA, 'keyword_norm_to_idf.pickle'), 'rb')
KEYWORD_NORM_TO_IDF = pickle.load(handle_keywords)


def is_equal_num(n1, n2):
    """Compare two numbers. If both 'exist' then return 1 if they are equal and 0 otherwise. Otherwise return -1."""
    if n1 is None or n2 is None:
        return 'missing'
    elif n1 == n2:
        return 'true'
    else:
        return 'false'


def is_contained_set(s1, s2):
    """Compare two sets. If both 'exist' then return 1 if they are equal and 0 otherwise. Otherwise return -1."""
    if is_missing(s1) or is_missing(s2):
        return 'missing'
    elif bool(s1 & s2):
        return 'true'
    else:
        return 'false'


def is_intersection_nonempty(s1, s2):
    if s1 != set() and s2 != set():
        if s1.intersection(s2) == set():
            return 'false'
        else:
            return 'true'
    elif s1 == set() and s2 == set():
        return 'both-missing'
    else:
        return 'one-missing'


def keep_value(i, j):
    assert i == j
    """Used to compare two identical values"""
    return i


def is_equal(s1, s2):
    """Compare two strings. If both 'exist' then return 1 if they are equal and 0 otherwise. Otherwise return -1."""
    if is_missing(s1) or is_missing(s2):
        return 'part-missing'
    elif s1 == s2:
        return 'true'
    else:
        return 'false'


def is_name_variation(s1, s2, name_to_nickname=NAME_TO_NICKNAME, name_variations=NAME_VARIATIONS):
    if (s1 in name_to_nickname.keys() and s2 in name_to_nickname[s1]) or (
            s2 in name_to_nickname.keys() and s1 in name_to_nickname[s2]):
        return True
    elif len(set(name_variations[s1]).intersection(set(name_variations[s2]))) > 0:
        return True
    else:
        return False


def is_equal_or_nickname(s1, s2):
    """Compare two names and return a value reflecting whether they are equal, have a nickname relation or are unequal.
    Names with nicknames are given as a dictionary where the key is a string (name) and value is a list of nicknames."""
    if is_missing(s1) or is_missing(s2):
        return 'part-missing'
    elif s1 == s2:
        return 'true'
    # elif is_name_variation(s1, s2):
    #     return 'nickname'
    else:
        return 'false'


def are_name_parts_equal(s1, s2):
    """Compare two strings. If both 'exist' then return 'true' if they are equal and 'false' otherwise.
     If only one is missing return 'one-missing'; else return 'both-missing'."""
    if is_missing(s1) and is_missing(s2):
        return 'both-missing'
    elif (is_missing(s1) and not is_missing(s2)) or (is_missing(s2) and not is_missing(s1)):
        return 'one-missing'
    elif s1 == s2:
        return 'true'
    else:
        return 'false'


def are_middle_names_equal(s1, s2):
    """Compare two strings. If both 'exist' then return 'true' if they are equal and 'false' otherwise.
     If only one is missing return 'one-missing'; else return 'both-missing'."""
    if is_missing(s1) and is_missing(s2):
        return 'both-missing'
    elif (is_missing(s1) and not is_missing(s2)) or (is_missing(s2) and not is_missing(s1)):
        return 'one-missing'
    elif len(s1) == len(s2) == 1 and s1 == s2:
        return "initials_equal"
    elif (len(s1) == 1 and len(s2) > 1 and s2.startswith(s1)) or (len(s2) == 1 and len(s1) > 1 and s1.startswith(s2)):
        return "initials_contained"
    elif s1 == s2 and len(s1) > 1 and len(s2) > 1:
        return 'true'
    else:
        return 'false'



def author_signature(doc, ai):
    first = doc['first_name'] or ai.split('.')[1] or ''
    middle = (doc['middle_name'] or doc['middle_name_init']) or ''
    return first, middle


def is_signature_compatible(sign1, sign2):
    def is_name_part_compatible(n1, n2):

        if n1 == '' or n2 == '':
            return True
        elif len(n1) == 1:
            return n2[0] == n1
        elif len(n2) == 1:
            return n1[0] == n2
        else:
            return n1 == n2 # or is_name_variation(n1, n2)

    # Signatures are compatible if first and middle names are compatible
    if is_name_part_compatible(sign1[0], sign2[0]) and is_name_part_compatible(sign1[1], sign2[1]):
        return 'true'
    # Names that are split but can also be written together
    elif (sign1[0] + sign1[1] == sign2[0] and sign2[1] == '') or (sign2[0] + sign2[1] == sign1[0] and sign1[1] == ''):
        return 'true'
    else:
        return 'false'


def are_signatures_compatible(signatures_1, signatures_2):
    """signatures_1 and signatures_2 contain all signatures compatible with the
    signature in the first position of each of the lists. This function checks whether all
    matching signatures are pairwise compatible."""

    combs = list(product(signatures_1, signatures_2))
    for c in combs:
        if is_signature_compatible(c[0], c[1]) == "false":
            return 1
    return 0


def is_signature_contained(sign1, sign2):
    """First word in sign1 needs to be contained in the first word of sign2 and second word as well.
    Specialty of first word checking is that nicknames are accepted as well."""

    def is_name_part_contained(n1, n2, with_nickname=False):
        if n1 == '':
            return True
        elif n2 == '':  # and n1 != ''
            return False
        elif n1 == n2:
            return True
        elif with_nickname is True  and len(n1) < len(n2):  # is_nickname does not reflect which one is a nickname of the other
            return True
        else:
            return n2.startswith(n1[0]) and len(n1) == 1

    return is_name_part_contained(sign1[0], sign2[0], with_nickname=True) and is_name_part_contained(sign1[1], sign2[1])


def compute_maximum_signatures(signatures):
    """"""
    sign_contained = {s: len([w for w in signatures if is_signature_contained(s, w)]) for s in
                      signatures}  # number of signatures which contain s; if s is maximal then this should be 1
    #  print(sign_contained)
    max_signatures = [s for s in signatures if sign_contained[s] == 1]
    return max_signatures


def tokenize(s):
    if is_missing(s):
        return ['']
    else:
        return word_tokenize(s.lower())


def compute_idf_sum_of_intersection(s1, s2, word_to_idf, filler=MISSING_VAL_FILLER):
    """Compute sum of IDF scores of words in the intersection set. If intersection is empty, returns 0."""
    common_words = s1.intersection(s2)
    if common_words != set():
        return sum([word_to_idf[w] for w in common_words if w in word_to_idf.keys()])
    else:
        return filler


def compute_idf_sum_of_acronyms(s1, s2):
    return compute_idf_sum_of_intersection(s1, s2, ACRONYM_TO_IDF)


def compute_idf_sum_of_arxiv_class(s1, s2):
    return compute_idf_sum_of_intersection(s1, s2, ARXIV_CLASS_TO_IDF)


def compute_idf_sum_of_keywords(s1, s2):
    return compute_idf_sum_of_intersection(s1, s2, KEYWORD_NORM_TO_IDF)


def extract_arxiv_classes(s):
    pattern = r"[-:]"
    s_split = re.split(pattern, s)
    return set([item.strip().lower() for item in s_split])


def extract_keywords(s):
    s_split = s.split(":")
    s_split = [item.strip().lower() for item in s_split]
    s_split = set([item for item in s_split if item != '-'])
    return s_split


def compare_existence(b1, b2):
    """Compare two Boolean variables. If both 'exist' then return 'true' if they are equal and 'false' otherwise.
     If only one is missing return 'one-missing'; else return 'both-missing'."""
    if b1 & b2:  # both True
        return 'true'
    elif b1 | b2:
        return 'one-missing'
    else:
        return 'false'


def cosine_sim_doc2vec_abs(v1, v2, vec_for_empty_string):
    if (v1 == vec_for_empty_string).all() or (v2 == vec_for_empty_string).all():
        return MISSING_VAL_FILLER
    else:
        return 1 - np.round(cosine(v1, v2), 2)


def jaro_winkler_dist(x, y, filler=MISSING_VAL_FILLER):
    # jarowinkler = JaroWinkler()
    if x == '' or y == '':
        return filler
    else:
        # return jarowinkler.distance(x.lower(), y.lower())
        return jaro_winkler(x.lower(), y.lower())


def is_in_specific_journal_subset(x, y, journal_subset=[
    "the astrophysical journal",
    "the astronomical journal",
    "nature",
    "science",
    "monthly notices of the royal astronomical society",
    "astronomy and astrophysics",
    "nature astronomy"]):
    result = sum([i in journal_subset for i in (x, y)])
    if result == 2:
        return "true"
    else:
        return "false"


def contains_word(x, y, word):
    """Compute whether both x and y contain word"""

    result = sum([word in i for i in (x, y)])
    if result == 2:
        return 'true'
    else:
        return 'false'


def textual_distance(name1, name2, dist_func="jaro_winkler"):
    if dist_func == "jaro_winkler":
        return jaro_winkler_dist(name1, name2)
    elif dist_func == "levenshtein":  # TODO: this needs handling of missing values
        return editdistance.eval(name1, name2)


def phonetic_distance(name1, name2, sim_func="dmetaphone", filler=MISSING_VAL_FILLER):
    if name1 == '' or name2 == '':
        return filler
    else:
        if sim_func == "soundex":
            sound1, sound2 = phonetics.soundex(name1), phonetics.soundex(name2)
        elif sim_func == "nysiis":
            sound1, sound2 = phonetics.nysiis(name1), phonetics.nysiis(name2)
        elif sim_func == "metaphone":
            sound1, sound2 = phonetics.metaphone(name1), phonetics.metaphone(name2)
        else:  # sim_func == "dmetaphone"
            sound1, sound2 = phonetics.dmetaphone(name1), phonetics.dmetaphone(name2)
        return editdistance.eval(sound1, sound2)


def extract_journal(doc):
    journal = doc['journal'].strip().lstrip().lower()
    return journal


def aggregate_journals(doclist):
    """Takes a list of documents and returns a set with the journals"""
    journals = set([])
    for i, doc in doclist.iterrows():
        journals.add(extract_journal(doc))
    return journals


def extract_coauthors(doc, use_reduce_ai=True):
    coauthors = doc['author_list'].split(';') if ';' in doc['author_list'] else [doc['author_list']]
    coauthors = [extract_surname_and_forenames(c) for c in coauthors if c is not None]
    coauthors = [normalize_name(clean_name_parts(c[1]), clean_name_parts(c[0])) for c in coauthors]
    # Remove author from list of coauthors
    au = extract_surname_and_forenames(doc['author'])
    author = normalize_name(clean_name_parts(au[1]), clean_name_parts(au[0]))
    if use_reduce_ai:
        coauthors = set([reduce_ai(c) for c in coauthors if c != author])
    else:
        coauthors = set([c for c in coauthors if c != author])
    return {c for c in coauthors if c is not None}