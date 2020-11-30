from os.path import join
from sklearn.experimental import enable_hist_gradient_boosting # needs to stay inside

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, average_precision_score, \
    fbeta_score
from sklearn.tree import DecisionTreeClassifier

from config import DIR_PATH

PATH_TO_D2V_MODEL_ABSTRACTS = join(DIR_PATH, 'models', 'd2v_abstracts_nostopwords.model')
PATH_TO_D2V_MODEL_TITLES = join(DIR_PATH, 'models', 'd2v_titles_nostopwords.model')


MODEL_FEATURES = (
    'aid_equal',
    'signature_match',
    'first_name_equal',
    'middle_name_equal',
    'middle_name_init_equal',
    'first_name_textual_dist',
    'first_name_phonetic_dist',
    # 'all_sig_compatible',
    'block_size',
    # 'block_signature_diversity',
    'coauthors_match',
    'age_diff',
    'age_max',
    'journal_equal',
    'journal_astro',
    'journal_astron',
    'journal_astrop',
    'journal_top',
    'journal_geo',
    # 'arxiv_class_bool_match',
    # 'arxiv_class_match',
    'keywords_bool_match',
    'keywords_match',
    'acronyms_bool_match',
    'acronyms_match',
    'top_words_match',
    # 'top_ngrams_match',
    'top_bigrams_match',
    'country_equal',
    'city_equal',
    'aff_bool_match',
    'aff_sim',
    'title_sim',
    'abstract_sim',
    'abstract_bool_match',
    'sig_comp_w_incomp_sigs',
    'last_name_chinese'
)

MAX_SIZE_TRAIN_DATA = 1000000  #

CLASSIFIERS = {"rf": RandomForestClassifier, "dt": DecisionTreeClassifier,
               "hist_gb": HistGradientBoostingClassifier}
GRAPH_CLUSTERINGS = ('label_prop', 'connected_components')

AI_BLOCKS_CLUSTERING = [
    'ables.s', 'adam.m', 'annibali.f', 'aramo.c', 'botvina.a', 'campbell-brown.m', 'carpenter.j',
    'cotzomi.j', 'desiante.r', 'ermakov.s', 'gretskov.p', 'katgert.p', 'klein.u', 'kuga.k',
    'luthcke.s', 'melioli.c', 'mendez.r', 'mercuri.s', 'miller.v', 'moreno.c', 'morlok.a', 'naef.d',
    'nestorov.g', 'phan.n', 'ranjan.s', 'ravelo.a', 'ribeiro.a', 'rovira.m', 'roy.b', 'salama.f',
    'sarasso.m', 'vargas.c', 'vincent.m',
    'binzel.r', 'lee.j',  'zhang.y', 'wang.j', 'russell.c', 'chen.y',
]


PARAM_GRID = {
    HistGradientBoostingClassifier: {
        'learning_rate': [0.0005, 0.001, 0.01, 0.05, 0.15, 0.3],
        'loss': ['binary_crossentropy'],
        'max_depth': [3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70],
        'l2_regularization': [0, 0.5, 1, 1.5, 3, 5],
        'max_iter': [20, 30, 50, 70, 100, 150, 200, 250, 300, 500],
        'min_samples_leaf': [3, 5, 10, 15, 20, 25, 30, 40, 50, 70],
    },
    RandomForestClassifier: {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [20, 30, 50, 70, 100, 150],
        'max_depth': range(5, 30, 5),
        # 'class_weight': CLASS_WEIGHTS,
        'oob_score': [True],
        'n_jobs': [-1],
        'min_samples_leaf': [5, 10, 15, 20, 25, 30, 40, 50]
    },
    DecisionTreeClassifier: {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(5, 30, 5),
        'min_samples_leaf': [5, 10, 15, 20, 25, 30, 40, 50],
        'max_features': ["auto", "sqrt", "log2"],
        # 'class_weight': CLASS_WEIGHTS,
        'random_state': [123]
    }
}
SCORERS = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'average_precision_score': make_scorer(average_precision_score),
    'fbeta_score': make_scorer(fbeta_score, beta=2)
}
