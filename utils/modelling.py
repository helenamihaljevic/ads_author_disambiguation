from os.path import join

import joblib
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.utils import resample

from config import DIR_PATH, PATH_MATRICES
from disambiguator import Disambiguator, TARGET, CATEGORICAL_FEATURES
from utils.helpers import extract_file_name_from_path, convert_str_to_list, load_pickle_file


def convert_binary_to_triple_schema(assignment):
    if assignment is True:
        return 1
    elif assignment is False:
        return -1
    else:
        return 0


def downsample(x, y, max_skewness=1, max_size=None, return_index=False):
    """Adopted from https://elitedatascience.com/imbalanced-classes"""
    df = x.copy()
    df.loc[:, 'TARGET'] = y
    df_class_1 = df[(df['TARGET'] == "true") | (df['TARGET'] == 1)]
    df_class_0 = df[(df['TARGET'] == "false") | (df['TARGET'] == 0)]

    # Downsample class 1
    if max_skewness == 0:  # change nothing
        size = len(df_class_1)
    elif max_skewness >= 1:
        size = int(min(max_skewness * len(df_class_0), len(df_class_1)))
    else:  # max_skewness < 1 but > 0, then increase reduce size of class 1 even more so that class 0 is overrepresented
        size = int(max_skewness * len(df_class_0))
    try:
        df_class_1_downsampled = resample(df_class_1, replace=False, n_samples=size, random_state=1)
    except ValueError:
        print("Class 0 still too large, smapling class 1 with replacement")
        df_class_1_downsampled = resample(df_class_1, replace=True, n_samples=size, random_state=1)

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_class_1_downsampled, df_class_0])
    if max_size and max_size < len(df_downsampled):
        df_downsampled = df_downsampled.sample(max_size, random_state=1)

    if return_index is True:
        return df_downsampled.index
    else:
        return df_downsampled.drop('TARGET', axis=1), df_downsampled['TARGET']

def export_decisiontree(tree_model, file_path, feature_names):
    export_graphviz(tree_model, out_file=file_path, precision=2,
                    filled=True, feature_names=feature_names)


def load_ai_blocks(files, max_size, min_size):
    blocks = []
    print(len(files))
    for f in files:
        df = pd.read_excel(f)
        if max_size >= len(df) > min_size:
            blocks.append(extract_file_name_from_path(f)[0])
    return blocks


def build_similarity_matrices(blocks, features, verbose=False, squared=False):
    """for each ai in blocks, build a DataFrame containing the attributes and a DataFrame containing all
     pairwise combinations (=features) of all authorships in the block.
     Return dictionaries {ai: attributes_df} and {ai: features_df}"""

    def build_similarity_matrix(block, features, verbose=False, squared=False):
        d = Disambiguator(block)
        """get attrs from features"""
        feature_to_attr = d.feature_to_pw_comp_method
        create_attrs = [v[0] for f, v in feature_to_attr.items() if f in features and f != "aid_equal"]
        create_attrs = list(set(create_attrs))
        print(f"create attrs: {create_attrs}")
        d.populate_block(verbose=verbose)
        for col in ['country', 'city']:  # entries are list objects but stored in file as strings
            if col in d.block.columns:
                d.block.loc[:, col] = d.block.loc[:, col].map(convert_str_to_list)
        d.ignore_ambiguous_authorships()
        d.prepare_attributes_for_modelling(create_attrs=create_attrs)
        if verbose:
            print("Attributes: ", d.attributes.columns)

        d.build_all_pw_comparisons(store_pw_combs=True, squared=squared, features=features)
        feats = d.feature_matrix.merge(d.pw_combs[['document_id_x', 'document_id_y']], how='outer',
                                       left_index=True,
                                       right_index=True)  # TODO: this simply adds document_ids and should be replaced by real IDs

        return feats, d.attributes

    print(f"features: {features}")
    ai_block_to_similarity_matrix = {}
    ai_block_to_all_attrs = {}
    print('...')
    print(len(blocks))

    for ai_block in blocks:
        if verbose:
            print(ai_block)
        feats, attrs = build_similarity_matrix(ai_block, features, verbose, squared)

        ai_block_to_all_attrs[ai_block] = attrs
        ai_block_to_similarity_matrix[ai_block] = feats

    return ai_block_to_similarity_matrix, ai_block_to_all_attrs


def remove_document_id_col(df):
    cols = [c for c in df.columns if not c.startswith('document_id')]
    return df[cols].copy()


def remove_duplicate_rows(df):
    return df.drop_duplicates().reset_index(drop=True)


def split_into_features_and_target(df, target_col):
    d = df.copy()
    d.loc[:, target_col] = d[target_col].map({'true': 1, 'false': 0})
    features_df, target = d.drop(target_col, axis=1), d[target_col]
    return features_df, target


def vectorize_categorical_features(df, target_col, cat_attrs):
    cat_cols = [c for c in df.columns if c in cat_attrs and c != target_col]
    return pd.get_dummies(df, columns=cat_cols)

def encode_path_params_for_model_dump(model_type, scoring_func, max_block_size, min_block_size, downsample_ratio):
    file_name_part = '_'.join([model_type, scoring_func, 'blocksize_range', str(min_block_size), str(max_block_size),
                               'max_skewness', str(downsample_ratio)])
    file_name_clf = file_name_part + '.pkl'
    file_name_validation_blocks = 'ai_blocks_validate_' + file_name_part + '.pkl'
    file_name_feature_info = 'feature_info_' + file_name_part + '.pkl'

    return file_name_clf, file_name_validation_blocks, file_name_feature_info


def load_model(model_type, scoring_func, max_block_size, min_block_size, downsample_ratio):
    file_name_clf, file_name_validation_blocks, \
    file_name_feature_info = encode_path_params_for_model_dump(model_type, scoring_func, max_block_size, min_block_size,
                                                               downsample_ratio)
    clf = joblib.load(join(DIR_PATH, 'models', file_name_clf))
    feature_info = joblib.load(join(DIR_PATH, 'models', file_name_feature_info))

    return clf, feature_info  # , ai_blocks_validate


def remove_obvious_nonmatches(df):
    """Remove rows from feature matrix (each row represents comparison of two authorships) where
    first_name or middle_name or middle_name_init are incompatible"""
    cond_first_name = (df['first_name_equal'] != 'false')
    cond_middle_name = (df['middle_name_equal'] != 'false')
    cond_middle_name_init = (df['middle_name_init_equal'] != 'false')

    return df[cond_first_name & cond_middle_name & cond_middle_name_init]


def filter_by_ai_blocks(block_to_f_mat, block_to_a_mat, blocks):
    block_to_f_mat_reduced = {k: v for k, v in block_to_f_mat.items() if k in blocks}
    block_to_a_mat_reduced = {k: v for k, v in block_to_a_mat.items() if k in blocks}

    return blocks, block_to_f_mat_reduced, block_to_a_mat_reduced


def fill_missing_feature_cols(df, feature_cols):
    for f in set(feature_cols).difference(set(df.columns)):
        df[f] = 0
    df = df.reindex(columns=feature_cols)
    return df


def filter_ai_blocks_by_size(block_to_matrix, block_to_attr, max_size, min_size):
    blocks = [b for b, m in block_to_attr.items() if max_size >= len(m) > min_size]
    bm = {b: m for b, m in block_to_matrix.items() if b in blocks}
    ba = {b: a for b, a in block_to_attr.items() if b in blocks}

    return blocks, bm, ba


def remove_au_pairs_with_same_doc_id(feat_matrix):
    doc_id_cols = [c for c in feat_matrix.columns if c.startswith('document_id')]
    assert len(doc_id_cols) == 2
    col_1, col_2 = doc_id_cols[0], doc_id_cols[1]
    """
    document_id cols contain author position after + sign.
    Restrict to only those rows (authorship pairs) where doc id is different.
    """
    return feat_matrix[
        feat_matrix[col_1].map(lambda x: x.split('+')[0]) != feat_matrix[col_2].map(
            lambda x: x.split('+')[0])].reset_index(drop=True)


def prepare_sim_matrix_for_clustering(feat_matrix, features):
    """Does not reduce the x-dimension of matrix, only manipulation of features (columns)"""
    data_cluster = remove_document_id_col(feat_matrix)
    x = split_into_features_and_target(data_cluster, TARGET)[0]
    x = vectorize_categorical_features(x, TARGET, CATEGORICAL_FEATURES)
    x = fill_missing_feature_cols(x, features)
    return x


def load_matrices(blocks):
    ai_block_to_matrix = load_pickle_file(join(PATH_MATRICES, 'ai_block_to_matrix.pickle'))
    ai_block_to_attrs = load_pickle_file(join(PATH_MATRICES, 'ai_block_to_attrs.pickle'))
    ai_blocks, ai_block_to_matrix, ai_block_to_attrs = filter_by_ai_blocks(ai_block_to_matrix, ai_block_to_attrs,
                                                                           blocks)

    return ai_block_to_attrs, ai_block_to_matrix
