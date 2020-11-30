import argparse
import datetime as dt
import random
import sys
from os.path import join

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

sys.path.append('../')
from config import PATH_MODELS, EXCEL_FILES, DIR_PATH
from disambiguator import NUMERICAL_FEATURES, ALL_FEATURES, TARGET, CATEGORICAL_FEATURES
from modelling_config import MODEL_FEATURES, CLASSIFIERS, AI_BLOCKS_CLUSTERING, PARAM_GRID, SCORERS, MAX_SIZE_TRAIN_DATA
from utils.modelling import downsample, load_ai_blocks, build_similarity_matrices, remove_document_id_col, \
    split_into_features_and_target, vectorize_categorical_features, remove_duplicate_rows, \
    encode_path_params_for_model_dump, filter_by_ai_blocks, filter_ai_blocks_by_size
from utils.helpers import convert_str_to_list, load_pickle_file, dump_pickle_file

parser = argparse.ArgumentParser(description='Train a classifier on pairs of authorships')
parser.add_argument('--final', type=bool, required=False, default=False,
                    help='if True then train model on all data, including profiles for clustering evaluation')
parser.add_argument('--n_iter', type=int, required=False, default=100,
                    help='Number of iterations for randomized search of optimal hyperparameters')
parser.add_argument('--cv', type=float, required=False, default=5,
                    help='Number of splits for cross validation')
parser.add_argument('--min_block_size', type=int, required=False, default=2,
                    help='Minimum block size')
parser.add_argument('--max_block_size', type=int, required=False, default=10000,
                    help='Maximum block size. if you want to use all, set to number > 1000')
parser.add_argument('--max_size_train_data', type=int, required=False, default=MAX_SIZE_TRAIN_DATA,
                    help='Maximum number of training samples to use. The higher the number, the slower the training.')

parser.add_argument('--model', type=str, required=False, default="rf", choices=("rf", "hist_gb", "dt"),
                    help='classification method. Choices correspond to Random Forest (rf), '
                         'Histogram Gradient Boosting (hist_gb), and Decision Tree (dt)')
parser.add_argument('--class_balance', required=False, default=1.0,
                    help='Whether to force certain ratio between both classes. '
                         'Set to 0 if you do not want to change ratios, 1 if classes should be balanced, > 1 '
                         'if class 1 is to be overrepresented and to a number between 0 and 1 for opposite.')
parser.add_argument('--scorer', required=False, default='average_precision_score', choices=SCORERS.keys(),
                    help='Scoring function used for optimization.')

args = parser.parse_args()

FEATURES_CATEGORY = "all"

PATH_CLUSTER_EVALS = join(PATH_MODELS, '_'.join(['cluster_evals', 'blocksize_range', str(args.min_block_size),
                                                 str(args.max_block_size), dt.date.today().strftime("%d%m%y"),
                                                 'testaff.pickle']))


def concat_similarity_matrices(ai_block_to_sim_matrix):
    sim_matrix_concat = pd.DataFrame()
    for sim_matrix in ai_block_to_sim_matrix.values():
        sim_matrix_concat = sim_matrix_concat.append(sim_matrix, ignore_index=True)
    return sim_matrix_concat


def reduce_squared_matrix_to_upper_triangle(df):
    """Assert to have columns document_id_x and document_id_y"""
    document_id_pairs = []

    for i, row in df.iterrows():
        new_pair = [row['document_id_x'], row['document_id_y']]
        if [row['document_id_x'], row['document_id_y']] not in document_id_pairs \
                and [row['document_id_y'], row['document_id_x']] not in document_id_pairs:
            document_id_pairs.append(new_pair)

    df_reduced = df.apply(lambda x: [x['document_id_x'], x['document_id_x']] in document_id_pairs, axis=1)
    print("Data size before reduction: ", len(df), "Data size after reduction: ", len(df_reduced))

    return df_reduced


def fill_missing_feature_cols(df, feature_cols):
    for f in set(feature_cols).difference(set(df.columns)):
        df[f] = 0
    df = df.reindex(columns=feature_cols)
    return df


def tune_hyperparams(model_type, refit_score, x, y, cv=args.cv, n_iter=args.n_iter,
                     tuning_method=RandomizedSearchCV):
    rs = tuning_method(model_type(random_state=1),
                       param_distributions=PARAM_GRID[model_type],
                       n_iter=n_iter,
                       cv=cv,
                       scoring=SCORERS,
                       refit=refit_score,
                       return_train_score=True, random_state=123, n_jobs=-1)

    rs.fit(x, y)
    print("Hyperparameter optimization done")
    return rs


def print_metrics(y, y_pred, round=False):
    if round:
        y_pred = y_pred.round()

    print('Accuracy: %.3f%%' % (accuracy_score(y, y_pred) * 100))
    # print('Precision: %.3f%%' % (precision_score(y, y_pred) * 100))
    # print('Recall: %.3f%%' % (recall_score(y, y_pred) * 100))
    # print('F1-score: %.3f%%' % (f1_score(y, y_pred) * 100))
    print('Average precision: %.3f%%' % (average_precision_score(y, y_pred) * 100))
    print(pd.DataFrame(confusion_matrix(y, y_pred),
                       columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    print('-----------')


def build_X_and_y(sim_matrix_classify):
    """Pre-process data for training of classifier"""

    doc_ids = sim_matrix_classify[[c for c in sim_matrix_classify.columns if c.startswith('document_id')]]
    sim_matrix_classify = remove_document_id_col(sim_matrix_classify)
    print(sim_matrix_classify.columns)
    sim_matrix_classify = remove_duplicate_rows(sim_matrix_classify)  # remove duplicate rows
    print("aid_equal value counts after removing duplicate entries: \n", sim_matrix_classify.aid_equal.value_counts())
    print(len(sim_matrix_classify))

    sim_matrix_classify = vectorize_categorical_features(sim_matrix_classify, TARGET, CATEGORICAL_FEATURES)
    x, y = split_into_features_and_target(sim_matrix_classify, TARGET)
    print("Overall distribution of target values: \n", y.value_counts())
    x = x.drop('ai_block', axis=1)
    return sim_matrix_classify, x, y, doc_ids


def partition(list_in, n):
    """split ai_blocks randomly into args.cv equally sized lists"""
    # https://stackoverflow.com/questions/3352737/python-randomly-partition-a-list-into-n-nearly-equal-parts
    random.seed(55)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def custom_cv_split(block_partition, sim_matrix, max_skewness):
    for part in block_partition:
        sim_matrix_train = sim_matrix[~sim_matrix['ai_block'].isin(part)]
        sim_matrix_test = sim_matrix[sim_matrix['ai_block'].isin(part)]

        train_index = downsample(sim_matrix_train.drop(TARGET, axis=1), sim_matrix_train[TARGET],
                                 max_skewness=max_skewness, max_size=args.max_size_train_data, return_index=True)
        test_index = downsample(sim_matrix_test.drop(TARGET, axis=1), sim_matrix_test[TARGET],
                                max_skewness=0, max_size=args.max_size_train_data, return_index=True)
        print("Size of training data: ", len(train_index))
        print("Size of test data: ", len(test_index))

        yield train_index, test_index


def dump_model(model, features_vectorized, scoring_func, max_block_size, min_block_size, downsample_ratio):
    model_type = str(model.__class__.__name__)
    file_name_clf, file_name_validation_blocks, \
    file_name_feature_info = encode_path_params_for_model_dump(model_type, scoring_func, max_block_size,
                                                               min_block_size,
                                                               downsample_ratio)
    joblib.dump(model, join(DIR_PATH, 'models', file_name_clf))

    feature_info = {
        'target': TARGET,
        'attrs': ALL_FEATURES,
        'num_attrs': NUMERICAL_FEATURES,
        'cat_attrs': CATEGORICAL_FEATURES,
        'features': features_vectorized
    }
    dump_pickle_file(join(PATH_MODELS, file_name_feature_info), feature_info)


if __name__ == '__main__':
    try:  # load precomputed matrices
        print("try")
        ai_block_to_matrix = load_pickle_file(join(DIR_PATH, 'matrices', 'ai_block_to_matrix.pickle'))
        ai_block_to_attrs = load_pickle_file(join(DIR_PATH, 'matrices', 'ai_block_to_attrs.pickle'))
        ai_blocks, ai_block_to_matrix, ai_block_to_attrs = filter_ai_blocks_by_size(ai_block_to_matrix,
                                                                                    ai_block_to_attrs,
                                                                                    args.max_block_size,
                                                                                    args.min_block_size)
        if args.final is False:
            ai_blocks_class = [b for b in ai_blocks if b not in AI_BLOCKS_CLUSTERING]
            ai_blocks, ai_block_to_matrix, ai_block_to_attrs = filter_by_ai_blocks(ai_block_to_matrix,
                                                                                   ai_block_to_attrs, ai_blocks_class)
    except FileNotFoundError:  # compute the matrices
        print("except")
        ai_blocks = load_ai_blocks(EXCEL_FILES, args.max_block_size, args.min_block_size)
        for col in ['country', 'city']:
            ai_blocks[col] = ai_blocks[col].map(convert_str_to_list)
        ai_block_to_matrix, ai_block_to_attrs = build_similarity_matrices(ai_blocks, ALL_FEATURES, verbose=False)

    print(len(ai_blocks))

    """Reduce features to MODEL_FEATURES and add ai_block as column"""
    for k, v in ai_block_to_matrix.items():
        if MODEL_FEATURES:
            ai_block_to_matrix[k] = v[list(MODEL_FEATURES)]
        ai_block_to_matrix[k]['ai_block'] = k

    print(f"Total number of ai_blocks: {len(ai_blocks)}")

    """concat all block-wise similarity matrices into one"""
    sim_matrix = concat_similarity_matrices(
        {k: v for k, v in ai_block_to_matrix.items() if k in ai_blocks})
    print(len(sim_matrix))
    print("aid_equal value counts at start: \n", sim_matrix.aid_equal.value_counts())

    """vectorize etc."""
    sim_matrix, X, Y, doc_ids = build_X_and_y(sim_matrix)
    print("vectorized features:\n ", X.columns)

    ai_block_partition = partition(ai_blocks, args.cv)

    """Hyperparameter Search for best classifier"""
    model = CLASSIFIERS[args.model]
    scorer = args.scorer
    cb = float(args.class_balance)

    print("Model Type: ", str(model.__name__))
    print("Scoring Func: ", scorer)
    print("Maximum skewness:", str(cb))

    custom_cv = custom_cv_split(ai_block_partition, sim_matrix, max_skewness=cb)
    result_rs = tune_hyperparams(model, scorer, X, Y, cv=custom_cv)

    print(f"Best params: {result_rs.best_params_}")
    print(f"Best score: {result_rs.best_score_}")

    eval_table = pd.DataFrame(result_rs.cv_results_)
    eval_table_file_name = '_'.join(['eval_table', str(model.__name__), scorer, str(cb), FEATURES_CATEGORY])
    eval_table.to_csv(join(DIR_PATH, 'model_evaluation', eval_table_file_name + '.csv'))

    clf_optimal = model(**result_rs.best_params_)
    x, y = downsample(X, Y, max_skewness=cb, max_size=args.max_size_train_data)
    print("distribution of target values: ", y.value_counts())
    clf_optimal.fit(x, y)

    dump_model(clf_optimal, X.columns, scorer, args.max_block_size, args.min_block_size, cb)
