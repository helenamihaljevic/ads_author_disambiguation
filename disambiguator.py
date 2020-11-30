from itertools import product, combinations
from os.path import join

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

from config import DIR_PATH, DIR_LABELED_DATA
from modelling_config import PATH_TO_D2V_MODEL_ABSTRACTS, PATH_TO_D2V_MODEL_TITLES
from utils.helpers import convert_to_set, is_missing, compute_diff, cosine_sim
from utils.pipeline import fetch_author_block, add_coauthors, is_equal, is_equal_num, \
    is_contained_set, author_signature, is_signature_compatible, are_name_parts_equal, keep_value, \
    tokenize, fetch_block_sizes, is_intersection_nonempty, is_equal_or_nickname, compute_idf_sum_of_acronyms, \
    extract_arxiv_classes, compute_idf_sum_of_arxiv_class, compare_existence, cosine_sim_doc2vec_abs, jaro_winkler_dist, \
    is_in_specific_journal_subset, contains_word, are_middle_names_equal, textual_distance, phonetic_distance, \
    extract_keywords, compute_idf_sum_of_keywords, are_signatures_compatible, extract_coauthors
from utils.text_proc import extract_name_parts, extract_acronyms, extract_middle_name_initial
from utils.tf_idf import create_corpus_from_docs, create_corpus_from_docs_keywords, \
    extract_top_keywords, extract_top_ngrams


class Disambiguator():
    # TODO: experimental; should be extended by other ethnicities
    chinese_surnames = ['wang', 'li', 'zhang', 'liu', 'chen', 'yang', 'huang', 'zhao', 'wu', 'zhou', 'xu', 'sun', 'ma',
                        'zhu', 'hu', 'guo', 'he', 'lin', 'gao', 'luo', 'zheng', 'liang', 'xie', 'song', 'tang', 'xu',
                        'deng', 'han', 'feng', 'cao', 'peng', 'zeng', 'xiao', 'tian', 'dong', 'pan', 'yuan', 'cai',
                        'jiang', 'yu', 'yu', 'du', 'ye', 'cheng', 'wei', 'su', 'lu', 'ding', 'ren', 'lu', 'yao', 'shen',
                        'zhong', 'jiang', 'cui', 'tan', 'lu', 'fan', 'wang', 'liao', 'shi', 'jin', 'wei', 'jia', 'xia',
                        'fu', 'fang', 'zou', 'xiong', 'bai', 'meng', 'qin', 'qiu', 'hou', 'jiang', 'yin', 'xue', 'yan',
                        'duan', 'lei', 'long', 'li', 'shi', 'tao', 'he', 'mao', 'hao', 'gu', 'gong', 'shao', 'wan',
                        'qin', 'wu', 'qian', 'dai', 'yan', 'ou', 'mo', 'kong', 'xiang', 'chang']

    DOC2VEC_ABSTRACTS = Doc2Vec.load(PATH_TO_D2V_MODEL_ABSTRACTS)
    DOC2VEC_TITLES = Doc2Vec.load(PATH_TO_D2V_MODEL_TITLES)
    DOC2VEC_ABSTRACTS_EMPTY_VEC = DOC2VEC_TITLES.infer_vector([''])

    attr_to_attr_creation_method = {
        'signature': '_prepare_signature',
        'first_name': '_prepare_name_parts',
        'middle_name': '_prepare_name_parts',
        'middle_name_init': '_prepare_name_parts',
        'last_name': '_prepare_name_parts',
        'journal': '_prepare_journal',
        'aff_org': '_prepare_aff_ner',
        'aff_loc': '_prepare_aff_ner',
        'aff': '_prepare_aff',
        'country': '_prepare_country',
        'city': '_prepare_city',
        'coauthors': '_prepare_coauthors',
        'top_words': '_prepare_top_keywords',
        'top_ngrams': '_prepare_top_ngrams',
        'top_bigrams': '_prepare_top_bigrams',
        'abstract_doc2vec': '_prepare_doc2vec_abstract',
        'title_doc2vec': '_prepare_doc2vec_title',
        'acronyms': '_prepare_acronyms',
        'block_size': '_prepare_signature_based_attributes',
        'block_signature_diversity': '_prepare_signature_based_attributes',
        'perc_matching_signatures': '_prepare_signature_based_attributes',
        'signature_frequency': '_prepare_signature_based_attributes',
        'arxiv_class': '_prepare_arxiv_class',
        'keywords': '_prepare_keywords',
        'age': '_prepare_age',
        'abstract_bool': '_prepare_abstract_bool',
        'arxiv_class_bool': '_prepare_arxiv_class_bool',
        'keywords_bool': '_prepare_keywords_bool',
        'aff_bool': '_prepare_aff_bool',
        'acronyms_bool': '_prepare_acronyms_bool',
        'all_sig_compatible': '_prepare_signature_based_attributes',
        'matching_signatures': '_prepare_signature_based_attributes'
    }

    feature_to_pw_comp_method = {
        'signature_match': ('signature', is_signature_compatible),
        'first_name_equal': ('first_name', is_equal_or_nickname),
        'first_name_textual_dist': ('first_name', lambda x, y: textual_distance(x, y, "jaro_winkler")),
        'first_name_phonetic_dist': ('first_name', lambda x, y: phonetic_distance(x, y, "dmetaphone")),
        # 'first_name_variant': ('first_name', ), # TODO: implement usage of synonym table
        'middle_name_equal': ('middle_name', are_middle_names_equal),
        'middle_name_init_equal': ('middle_name_init', are_name_parts_equal),
        'journal_equal': ('journal', is_equal),
        'journal_astro': ('journal', lambda x, y: contains_word(x, y, "astro")),
        'journal_astron': ('journal', lambda x, y: contains_word(x, y, "astron")),
        'journal_astrop': ('journal', lambda x, y: contains_word(x, y, "astrop")),
        'journal_geo': ('journal', lambda x, y: contains_word(x, y, "geo") or contains_word(x, y, "earth")),
        'journal_top': ('journal', is_in_specific_journal_subset),
        # 'orcid_id_equal': ('orcid_id', is_equal),
        'aff_sim': ('aff', jaro_winkler_dist),
        'country_equal': ('country', is_intersection_nonempty),
        'city_equal': ('city', is_intersection_nonempty),
        'coauthors_match': ('coauthors', is_contained_set),
        'top_words_match': ('top_words', is_contained_set),
        'top_ngrams_match': ('top_ngrams', is_contained_set),
        'top_bigrams_match': ('top_bigrams', is_contained_set),
        'aid_equal': ('aid', is_equal_num),
        'abstract_sim': (
            'abstract_doc2vec', lambda x, y: cosine_sim_doc2vec_abs(x, y, Disambiguator.DOC2VEC_ABSTRACTS_EMPTY_VEC)),
        'title_sim': ('title_doc2vec', cosine_sim),
        'acronyms_match': ('acronyms', compute_idf_sum_of_acronyms),
        'block_size': ('block_size', keep_value),
        'block_signature_diversity': ('block_signature_diversity', keep_value),
        'perc_matching_signatures': ('perc_matching_signatures', compute_diff),
        'signature_frequency': ('signature_frequency', compute_diff),
        'arxiv_class_match': ('arxiv_class', compute_idf_sum_of_arxiv_class),
        'keywords_match': ('keywords', compute_idf_sum_of_keywords),
        'age_max': ('age', lambda x, y: max(x, y)),
        'age_diff': ('age', compute_diff),
        'abstract_bool_match': ('abstract_bool', compare_existence),
        'arxiv_class_bool_match': ('arxiv_class_bool', compare_existence),
        'keywords_bool_match': ('keywords_bool', compare_existence),
        'aff_bool_match': ('aff_bool', compare_existence),
        'acronyms_bool_match': ('acronyms_bool', compare_existence),
        'all_sig_compatible': ('all_sig_compatible', keep_value),
        'sig_comp_w_incomp_sigs': ('matching_signatures', are_signatures_compatible),
        'last_name_chinese': ('last_name', lambda x, y: 1 if x in Disambiguator.chinese_surnames else 0)
    }

    target = 'aid_equal'
    all_features = tuple([a for a in feature_to_pw_comp_method.keys()])
    numerical_features = ['age_diff', 'block_size', 'abstract_sim', 'title_sim', 'block_signature_diversity',
                          'aff_sim', 'perc_matching_signatures', 'signature_frequency', 'acronyms_match',
                          'arxiv_class_match', 'keywords_match', 'age_max', 'first_name_textual_dist',
                          'first_name_phonetic_dist', 'sig_comp_w_incomp_sigs']  # TODO: last attribute in fact 0 or 1
    features_requiring_missing_value_imputation = ['acronyms_match', 'arxiv_class_match', 'abstract_sim', 'aff_sim',
                                                   'first_name_textual_dist', 'first_name_phonetic_dist']

    def __init__(self, ai):
        # e.g. ai = 'miller.j
        self.ai = ai
        # Stores document data as returned by db or file
        self.block = None
        # Stores features (preprocessed data)
        self.attributes = pd.DataFrame()
        # Stores pairwise comparisons of features
        self.feature_matrix = None
        self.pw_combs = None
        self.use_for_modelling = []

    def populate_block(self, source, verbose=False):
        if source == 'sql':
            if verbose:
                print('Populating block via SQL query against db')
            self.block = fetch_author_block(self.ai)
            self.block = add_coauthors(self.block)
            self.block.insert(loc=0, column='aid', value=None)
            self.block = self.block[self.block.columns[:]]
        elif source == 'file':
            if verbose:
                print('Populating block from file')
            datafile = join(DIR_PATH, DIR_LABELED_DATA, f'{self.ai}.xlsx')
            self.block = pd.read_excel(datafile)
        else:
            print('Allowed values for source are: sql, file')
            return
        # Reset index, sort by year, author
        self.block = self.block.sort_values(by=['year', 'author']).reset_index(drop=True)
        if verbose:
            print(f'Populated block with {len(self.block)} authorships')

    def ignore_ambiguous_authorships(self):
        self.block = self.block[pd.notnull(self.block['aid'])]

    def prepare_attributes_for_modelling(self, create_attrs, select_attrs=('document_id', 'year', 'aid')):
        self._select_attributes(list(select_attrs))
        # These are the block related attributes
        signature_attrs = ['signature', 'block_signature_diversity', 'matching_signatures',
                           'perc_matching_signatures', 'signature_frequency',
                           'block_size', 'all_sig_compatible']
        # These attributes can be created first independently of others
        first_attrs = [a for a in create_attrs if a not in signature_attrs]
        print("first attrs", first_attrs)
        # Make first the first attrs
        create_attrs_methods = set([Disambiguator.attr_to_attr_creation_method[a] for a in first_attrs])
        for m in create_attrs_methods:
            getattr(self, m)()

        # Create signature and block attributes afterwards (they need first_name, etc to exist!)
        # print(self.attributes.columns)

        if 'signature' in create_attrs:
            print("signature in create_attrss")
            self._prepare_signature()
            print(self.attributes.columns)

            if set(signature_attrs).issubset(set(create_attrs)):
                print("yes all sig attrs are there")
                self._prepare_signature_based_attributes()

        self._filter_attributes()

    def build_all_pw_comparisons(self, features, store_pw_combs=False, squared=False):

        use_attrs = [v[0] for k, v in Disambiguator.feature_to_pw_comp_method.items()]
        use_attrs = [attr for attr in self.attributes.columns if attr in use_attrs]

        # Store document_id to keep track of data origin later
        combs = Disambiguator.build_pw_comparisons(self.attributes[['document_id'] + use_attrs], squared=squared)
        filtered_attr_to_method = {f: v for f, v in Disambiguator.feature_to_pw_comp_method.items() if
                                   f in features and v[0] in use_attrs}

        self.feature_matrix = pd.DataFrame()
        for f, a_m in sorted(filtered_attr_to_method.items()):
            attr, method = a_m[0], a_m[1]
            self.feature_matrix[f] = combs.apply(lambda x: method(x[attr + '_x'], x[attr + '_y']), axis=1)

        if store_pw_combs:
            self.pw_combs = combs

    def _select_attributes(self, attr_names):
        for a in attr_names:
            self.attributes[a] = self.block[a]
        self.use_for_modelling.extend(attr_names)

    def _prepare_aff(self):
        self.attributes['aff'] = self.block['affiliation'].fillna("")
        self.use_for_modelling.append('aff')

    def _prepare_name_parts(self):
        self.attributes['last_name'] = self.block['author'].map(lambda x: x.split(",")[0]).unique()[0]
        if 'first_name' in self.attributes.columns and 'middle_name' in self.attributes.columns:
            self.attributes['middle_name_init'] = self.block['author'].map(extract_middle_name_initial)
            self.use_for_modelling.extend(['middle_name_init'])
        else:
            self.attributes['first_name'], self.attributes['middle_name'], self.attributes['middle_name_init'] = zip(
                *self.block['author'].map(extract_name_parts))

            self.use_for_modelling.extend(['last_name','first_name', 'middle_name', 'middle_name_init'])

    def _prepare_country(self):
        self.attributes['country'] = self.block['country'].map(convert_to_set)
        self.use_for_modelling.append('country')

    def _prepare_city(self):
        self.attributes['city'] = self.block['city'].map(convert_to_set)
        self.use_for_modelling.append('city')

    def _prepare_journal(self):
        self.attributes['journal'] = self.block['journal'].apply(str.lower)
        self.use_for_modelling.append('journal')

    def _prepare_top_keywords(self):
        corpus = create_corpus_from_docs_keywords(self.block)
        self.attributes['top_words'] = self.block.apply(lambda x: set([l[0] for l in extract_top_keywords(x, corpus)]),
                                                        axis=1)

        self.use_for_modelling.append('top_words')

    def _prepare_top_ngrams(self, ngram_range=(1, 2), lim=10):
        corpus = create_corpus_from_docs(self.block)
        self.attributes['top_ngrams'] = self.block.apply(lambda x: set(extract_top_ngrams(x, corpus,
                                                                                          ngram_range=ngram_range,
                                                                                          lim=lim)), axis=1)
        self.use_for_modelling.append('top_ngrams')

    def _prepare_top_bigrams(self, ngram_range=(2, 2), lim=10):
        corpus = create_corpus_from_docs(self.block)
        self.attributes['top_bigrams'] = self.block.apply(lambda x: set(extract_top_ngrams(x, corpus,
                                                                                           ngram_range=ngram_range,
                                                                                           lim=lim)), axis=1)
        self.use_for_modelling.append('top_bigrams')

    def _prepare_coauthors(self):
        self.attributes['coauthors'] = self.block.apply(extract_coauthors, axis=1)

        self.use_for_modelling.append('coauthors')

    def _prepare_signature(self):
        assert {'first_name', 'middle_name', 'middle_name_init'}.issubset(
            set(self.attributes.columns)), "Required columns not in attributes"
        self.attributes['signature'] = self.attributes.apply(lambda r: author_signature(r, self.ai), axis=1)

        """Some names occur written in one word or with a '-' connecting first and middle name. 
        If such a word occurs, then split it."""

        unique_signatures = self.attributes['signature'].unique()
        signatures_double_name = [s for s in unique_signatures if s[0] != '' and s[1] != '']
        if signatures_double_name != []:
            signatures_double_name = {s[0] + s[1]: s for s in
                                      signatures_double_name}  # not necessarily unique but should be extremely rare
            first_name_only = [s[0] for s in unique_signatures if s[0] != '' and s[1] == '']

            for s in first_name_only:
                if s in signatures_double_name.keys():
                    cond_name = self.attributes.signature == (s, '')
                    # print("Signature that could also be split into two parts: ", s)
                    first, middle = signatures_double_name[s][0], signatures_double_name[s][1]
                    print("first and middle: ", first, middle)
                    self.attributes.loc[cond_name, 'first_name'] = first
                    self.attributes.loc[cond_name, 'middle_name'] = middle
                    self.attributes.loc[cond_name, 'middle_name_init'] = middle[0]
                    same_cond = (self.attributes.first_name == first) & (self.attributes.middle_name == middle)

                    self.attributes.loc[same_cond, 'signature'] = self.attributes.loc[same_cond, :].apply(
                        lambda x: (x['first_name'], x['middle_name']), axis=1)

        self.use_for_modelling.append('signature')

    def _prepare_signature_based_attributes(self):
        # Requires to know all signatures present in the block
        assert ('signature') in self.attributes.columns, 'Required column signature not in attributes'

        block_size = len(self.block)
        unique_signatures = list(self.attributes['signature'].unique())

        # How many distinct (first, middle) signatures in block
        self.attributes['block_signature_diversity'] = np.round(len(unique_signatures) / block_size, 3)

        # list of all compatible signatures with s in first position
        self.attributes['matching_signatures'] = self.attributes.signature.map(
            lambda s: [s] + [s1 for s1 in [i for i in unique_signatures if i != s]
                             if is_signature_compatible(s, s1) == "true"])
        # With how many of those signatures each particular record matches
        # I.e. in block with (john s) and (jake simon) a pub with (j s) matches 2
        self.attributes['perc_matching_signatures'] = self.attributes.signature.apply(
            lambda s: sum([int(x == 'true') for x in [is_signature_compatible(s, s1) for s1 in unique_signatures]]))
        self.attributes['perc_matching_signatures'] = np.round(self.attributes['perc_matching_signatures'] / len(
            unique_signatures), 3)

        # How frequent the signature is
        # I.e. in block with (john s) = 3 and (jake simon) = 1, freques are 0.75 and 0.25
        count_sign = lambda s: self.attributes.signature.value_counts().loc[[s]].values[0]
        self.attributes['signature_frequency'] = self.attributes.signature.apply(
            count_sign) / block_size
        self.attributes['signature_frequency'] = self.attributes['signature_frequency'].map(lambda x: np.round(x, 3))

        # smallest quantile for which the block size value is larger than length of current block
        block_size_quantiles = fetch_block_sizes()
        self.attributes['block_size'] = min([q for q, v in block_size_quantiles.items() if v >= len(self.block)])

        unique_signatures = self.attributes["signature"].unique()
        unique_signature_pairs = set(combinations(unique_signatures, 2))
        incompatible_sig_pairs = [sig_pair for sig_pair in unique_signature_pairs if
                                  is_signature_compatible(*sig_pair) != 'true']
        print(incompatible_sig_pairs)

        if len(incompatible_sig_pairs) > 0:
            self.attributes['all_sig_compatible'] = 'false'
        else:
            self.attributes['all_sig_compatible'] = 'true'

        # self.use_for_modelling.extend(
        #     ['block_size', 'block_signature_diversity', 'perc_matching_signatures', 'signature_frequency', 'all_sig_compatible'])
        self.use_for_modelling.extend(
            ['block_size', 'block_signature_diversity', 'all_sig_compatible', 'matching_signatures'])


    def _prepare_doc2vec_abstract(self):
        self.attributes['abstract_doc2vec'] = self.block['abstract'].map(tokenize)
        self.attributes['abstract_doc2vec'] = self.attributes['abstract_doc2vec'].map(
            lambda x: Disambiguator.DOC2VEC_ABSTRACTS.infer_vector(x))
        self.use_for_modelling.append('abstract_doc2vec')

    def _prepare_doc2vec_title(self):
        self.attributes['title_doc2vec'] = self.block['title'].map(tokenize)
        self.attributes['title_doc2vec'] = self.attributes['title_doc2vec'].map(
            lambda x: Disambiguator.DOC2VEC_TITLES.infer_vector(x))
        self.use_for_modelling.append('title_doc2vec')

    def _prepare_acronyms(self):
        self.attributes['acronyms'] = self.block['abstract'].map(str) + ' ' + self.block['title']
        self.attributes['acronyms'] = self.attributes['acronyms'].map(extract_acronyms)

        self.use_for_modelling.append('acronyms')

    def _prepare_arxiv_class(self):

        self.attributes['arxiv_class'] = self.block['arxiv_class'].fillna('')
        self.attributes['arxiv_class'] = self.attributes['arxiv_class'].map(extract_arxiv_classes)
        self.attributes['arxiv_class'] = self.attributes['arxiv_class'].map(lambda x: set() if x == {''} else x)

        self.use_for_modelling.append('arxiv_class')

    def _prepare_keywords(self):
        self.attributes['keywords'] = self.block['keywords'].fillna('')
        self.attributes['keywords'] = self.attributes['keywords'].map(extract_keywords)
        self.attributes['keywords'] = self.attributes['keywords'].map(lambda x: set() if x == {''} else x)

        self.use_for_modelling.append('keywords')

    def _prepare_age(self):
        self.attributes['age'] = 2018 - self.block['year']  # TODO: replace 2018 by this year in production or more recent data
        self.use_for_modelling.append('age')

    def _prepare_abstract_bool(self):
        self.attributes['abstract_bool'] = self.block['abstract'].map(lambda x: not is_missing(x))
        self.use_for_modelling.append('abstract_bool')

    def _prepare_arxiv_class_bool(self):
        self.attributes['arxiv_class_bool'] = self.block['arxiv_class'].map(lambda x: not is_missing(x))
        self.use_for_modelling.append('arxiv_class_bool')

    def _prepare_keywords_bool(self):
        self.attributes['keywords_bool'] = self.block['keywords'].map(lambda x: not is_missing(x))
        self.use_for_modelling.append('keywords_bool')

    def _prepare_aff_bool(self):
        self.attributes['aff_bool'] = self.block['affiliation'].map(lambda x: not is_missing(x))
        self.use_for_modelling.append('aff_bool')

    def _prepare_acronyms_bool(self):
        # TODO: improve code later. first 2 lines are repeated from _prepare_acronyms
        self.attributes['acronyms_bool'] = self.block['abstract'].map(str) + ' ' + self.block['title']
        self.attributes['acronyms_bool'] = self.attributes['acronyms_bool'].map(extract_acronyms)
        self.attributes['acronyms_bool'] = self.attributes['acronyms_bool'].map(lambda x: len(x) > 0)
        self.use_for_modelling.append('acronyms_bool')

    def _filter_attributes(self):
        self.attributes = self.attributes[sorted(self.use_for_modelling)]

    @staticmethod
    def build_pw_comparisons(df, squared):
        if squared is True:
            combs = list(product(df.values.tolist(), repeat=2))
        else:
            combs = list(combinations(df.values.tolist(), 2))
        combs = [item[0] + item[1] for item in combs]  # flatten combs which is a list of tuples of len 2
        new_cols = [c + '_x' for c in df.columns] + [c + '_y' for c in df.columns]
        combs = pd.DataFrame(combs, columns=new_cols)
        return combs


TARGET = Disambiguator.target
ALL_FEATURES = Disambiguator.all_features
NUMERICAL_FEATURES = Disambiguator.numerical_features
CATEGORICAL_FEATURES = [a for a in Disambiguator.all_features if a not in NUMERICAL_FEATURES]
