import argparse
import sys
import time

import numpy as np

from disambiguator import ALL_FEATURES

sys.path.append('../')

from sklearn.metrics import classification_report
from modelling_config import *
from utils.modelling import load_model, prepare_sim_matrix_for_clustering, build_similarity_matrices, load_matrices, \
    remove_au_pairs_with_same_doc_id

import pandas as pd
from cluster import Clusterizer, Cluster

pd.set_option('display.max_columns', 500)

parser = argparse.ArgumentParser(
    description='Evaluate a trained classification model together with a clustering approach on a set of validation author blocks')
parser.add_argument('--min_block_size', type=int, required=False, default=2,
                    help='Minimum block size')
parser.add_argument('--max_block_size', type=int, required=False, default=10000,
                    help='Maximum block size. if you want to use all, set to number > 1000')
parser.add_argument('--model', type=str, required=False, default="rf", choices=("rf", "gb", "hist_gb", "dt"),
                    help='classification method. Choices correspond to Random Forest (rf), "Gradient Boosting (gb), '
                         'Histogram Gradient Boosting (hist_gb), and Decision Tree (dt)')
parser.add_argument('--class_balance', required=False, default=1,
                    help='Whether to force certain ration between both classes. '
                         'Set to 0 if you do not want to change ratios, 1 if classes should be balanced, and > 1 '
                         'will cause the majority class to have the corresponding maximal ration over the minority class.')
parser.add_argument('--scorer', required=False, default='average_precision_score', choices=SCORERS.keys(),
                    help='Scoring function used for optimization.')
parser.add_argument('--clustering', required=False, default='label_prop', choices=GRAPH_CLUSTERINGS,
                    help='Clustering algorithm')

args = parser.parse_args()


def construct_author_id(ai, num):
    return ai + '.' + str(num)


def main(clf, blocks, block_to_matrix, block_to_attrs, features):
    cluster_evaluation = pd.DataFrame(columns=['b3prec', 'b3recall', 'b3Fscore'])
    original_block_sizes = []
    disamb_block_sizes = []
    number_clusters = []
    number_clusters_pred = []
    ai_to_clusters = {}
    ai_to_classifier_evaluation = {}
    truth_all = {}
    clu_map_all = {}
    nodes_edges_times = []
    for ai in blocks:
        # print(ai)
        start_prep = time.time()
        attrs_matrix = block_to_attrs[ai]
        """block sizes"""
        number_clusters.append(attrs_matrix.aid.nunique())
        original_block_sizes.append(len(attrs_matrix))  # number of authorships
        disamb_block_sizes.append(
            len(attrs_matrix[pd.notnull(attrs_matrix.aid)]))  # number of disambiguated authorships
        sim_matrix = block_to_matrix[ai]
        sim_matrix = remove_au_pairs_with_same_doc_id(sim_matrix)
        x = prepare_sim_matrix_for_clustering(sim_matrix, features)
        finish_prep = time.time()

        """clustering"""
        clu = Clusterizer()
        start_clf = time.time()
        clu.apply_classifier(clf, x)
        """collect data to evaluate classifier on clustering ai_blocks"""
        classifier_true_predict = sim_matrix[['aid_equal']].copy()
        classifier_true_predict["aid_equal"] = classifier_true_predict["aid_equal"].map({'true': 1, 'false': 0})
        classifier_true_predict["predicted"] = clu.predictions["y_pred"]
        finish_clf = time.time()

        ai_to_classifier_evaluation[ai] = classifier_true_predict
        start_clu = time.time()
        clu.create_clusters(attrs_matrix, sim_matrix, verbose=False, components=args.clustering)
        clu.clusters = [Cluster(cluster_id=ai + '_' + str(int(c.id)), docs_ids=c.docs_ids) for c in
                        clu.clusters]
        number_clusters_pred.append(len(clu.clusters))

        ai_to_clusters[ai] = clu
        truth, clu_map = clu.compute_true_and_cluster_assignment(attrs_matrix)
        metrics = clu.compute_bcubed_metrics(truth, clu_map)
        finish_clu = time.time()

        cluster_evaluation = cluster_evaluation.append(pd.Series(metrics, name=ai))
        """Put all clusters for all blocks together and evaluate at once"""
        truth_all.update({k: {ai + '_' + str(int(item)) for item in v} for k, v in truth.items()})  # v are sets
        clu_map_all.update(clu_map)
        time_prep = np.round(finish_prep - start_prep, 3)
        time_clf = np.round(finish_clf - start_clf,3)
        time_clu = np.round(finish_clu - start_clu,3)
        nodes_edges_times.append([ai, clu.graph.number_of_nodes(),clu.graph.number_of_edges(),time_prep, time_clf, time_clu])

    cluster_evaluation['bs_orig'] = original_block_sizes
    cluster_evaluation['bs_disamb'] = disamb_block_sizes
    cluster_evaluation['n_clu'] = number_clusters
    cluster_evaluation['n_clu_pred'] = number_clusters_pred

    all_metrics = Clusterizer.compute_bcubed_metrics(truth_all, clu_map_all)  # considers all blocks together

    return ai_to_clusters, cluster_evaluation, ai_to_classifier_evaluation, all_metrics, nodes_edges_times


if __name__ == '__main__':
    start_data_stuff = time.perf_counter()
    try:
        ai_block_to_attrs, ai_block_to_matrix = load_matrices(blocks=AI_BLOCKS_CLUSTERING)
    except:
        ai_blocks = AI_BLOCKS_CLUSTERING
        ai_block_to_matrix, ai_block_to_attrs = build_similarity_matrices(ai_blocks, ALL_FEATURES, verbose=False)

    print(len(AI_BLOCKS_CLUSTERING))
    print(AI_BLOCKS_CLUSTERING)

    model = CLASSIFIERS[args.model]
    scorer = args.scorer
    try:
        cb = float(args.class_balance)
        clf, feature_info = load_model(str(model.__name__), scorer, args.max_block_size, args.min_block_size,
                                       cb)
    except:
        cb = int(args.class_balance)
        clf, feature_info = load_model(str(model.__name__), scorer, args.max_block_size, args.min_block_size,
                                       cb)

    print("Model Type: ", str(model.__name__))
    print("Scoring Func: ", scorer)
    print("Maximum skewness:", cb)

    features_vectorized = feature_info['features']

    ai_to_clu, clu_eval, clf_eval, overall_metrics, complexity = main(clf, AI_BLOCKS_CLUSTERING, ai_block_to_matrix,
                                                          ai_block_to_attrs, features_vectorized)
    print("-------------------------")
    """print classification metrics"""
    clf_eval_df = pd.concat(list(clf_eval.values()))
    print(classification_report(clf_eval_df["aid_equal"], clf_eval_df["predicted"]))
    print("-------------------------")

    """print clustering metrics"""
    clu_eval.rename(columns={"b3prec": "b3p", "b3recall": "b3r", "b3Fscore": "b3F"}, inplace=True)
    print(clu_eval.apply(lambda x: np.round(x, 3)))

    print("-------------------------")
    metric_eval = pd.DataFrame([clu_eval.mean()])
    metric_eval = metric_eval[["b3p", "b3r", "b3F", "n_clu", "n_clu_pred"]]
    metric_eval.rename(columns={"b3p": "b3p_mi", "b3r": "b3r_mi", "b3F": "b3F_mi"}, inplace=True)  # mi for micro
    metric_eval["b3p_ma"] = overall_metrics["b3prec"]  # ma for macro
    metric_eval["b3r_ma"] = overall_metrics["b3recall"]
    metric_eval["b3F_ma"] = overall_metrics["b3Fscore"]

    print(metric_eval[["b3p_mi", "b3r_mi", "b3F_mi", "b3p_ma", "b3r_ma", "b3F_ma", "n_clu", "n_clu_pred"]].apply(
        lambda x: np.round(x, 3)))
    print("-------------------------")
    """Nodes, Edges, time complexity"""
    complexity = pd.DataFrame(complexity, columns=["ai", "nodes", "edges","time_prep", "clf_time", "clu_time"])
    print(complexity)
    print("-------------------------")
