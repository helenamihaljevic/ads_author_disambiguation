import bcubed
import networkx
import pandas as pd
from networkx.algorithms.clique import find_cliques
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.components.connected import connected_components


class Cluster():

    def __init__(self, cluster_id, docs_ids=[], docs=[]):
        """ A cluster is initialized when 2 documents are decided to be from same author
        Or from a single document """
        self.id = cluster_id
        self.repr = None
        self.docs = docs
        self.docs_ids = docs_ids

    @property
    def size(self):
        return len(self.docs_ids)

    def add_to_cluster(self, doc_id):
        """ Add a new doc to an existing cluster
        If doc already in cluster, do nothing """
        self.docs_ids.append(doc_id)

    def remove_from_cluster(self, doc_id):
        self.docs_ids = [did for did in self.docs_ids if did != doc_id]


class Clusterizer():
    """ Creates a collection of clusters """

    def __init__(self):
        self.last_cluster_id = 0
        self.clusters = []
        self.predictions = None
        self.matched_documents = []
        self.components = []
        self.cliques = []
        self.metrics = None

    @property
    def number_of_clusters(self):
        return len(self.clusters)

    def apply_classifier(self, model, X):
        results = pd.DataFrame()
        results['y_pred'] = model.predict(X)
        results['y_pred_proba'] = [x[1] for x in model.predict_proba(X)]  # prob for class 1
        self.predictions = results

    @property
    def matches(self):
        return self.predictions[self.predictions['y_pred'] == 1]
        # return self.predictions[self.predictions['y_pred_proba'] >= 0.7] # if you want to have more/less edges based on proba

    def _get_matched_documents(self, keys, set_weight=False):
        for ind in self.matches.index:
            id1 = keys.iloc[ind]['document_id_x']
            id2 = keys.iloc[ind]['document_id_y']
            weight = self.predictions.iloc[ind]['y_pred_proba']
            if not set_weight:
                self.matched_documents.append((id1, id2, 1))
            else:
                self.matched_documents.append((id1, id2, weight))

    def get_cluster_by_id(self, cluster_id):
        return [c for c in self.clusters if c.id == cluster_id][0]

    def get_cluster_by_doc_id(self, doc_id):
        return [c for c in self.clusters if doc_id in c.docs_ids][0]

    def build_graph(self):
        G = networkx.Graph()
        G.add_weighted_edges_from(self.matched_documents)
        self.graph = G

    def create_clusters_from_components(self):
        self.clusters = [Cluster(cluster_id=i, docs_ids=ids) for i, ids in enumerate(self.components)]
        if len(self.clusters) > 0:
            self.last_cluster_id = max([clu.id for clu in self.clusters])
        else:
            self.last_cluster_id = 0

    def create_clusters_from_leftover_documents(self, attrs):
        """ Locates leftover documents and creates single-element clusters """
        set_all = set(attrs.document_id.unique())
        set_in_clusters = set([item for sublist in [c.docs_ids for c in self.clusters] for item in sublist])
        leftovers = [[id] for id in set_all - set_in_clusters]

        self.clusters.extend([Cluster(i + 1 + self.last_cluster_id, docs_ids=ids) for i, ids in enumerate(leftovers)])

        self.last_cluster_id = max([clu.id for clu in self.clusters])

    def create_clusters(self, attrs, keys, verbose=False, components='connected_components'):
        if components == 'label_prop':
            self._get_matched_documents(keys, set_weight=True)
        else:
            self._get_matched_documents(keys, set_weight=False)

        self.build_graph()

        if components == 'connected_components':
            self.components = [list(a) for a in list(connected_components(self.graph))]
        elif components == 'cliques':  # TODO: needs fixing of docs_ids belonging to more than one max.clique -> number_of_cliques > 1
            self.components = [list(a) for a in list(find_cliques(self.graph))]
        elif components == 'biconnected_components':
            self.components = [list(a) for a in list(networkx.biconnected_components(self.graph))]
        elif components == 'label_prop':
            self.components = [list(a) for a in list(asyn_lpa_communities(self.graph, weight='weight', seed=0))]

        self.create_clusters_from_components()
        self.create_clusters_from_leftover_documents(attrs)

        if components == 'cliques':
            all_ids = [c.docs_ids for c in self.clusters]
            all_ids = [item for sublist in all_ids for item in sublist]
            # print("all_ids:", all_ids)
            id_to_number_of_cliques = {i: networkx.number_of_cliques(self.graph, nodes=i) for i in all_ids}
            print("ids belonging to multiple cliques:")
            print({k: v for k, v in id_to_number_of_cliques.items() if v > 1})
        if verbose:
            print(f'Created {self.last_cluster_id} clusters')
            for clu in self.clusters:
                print(f'Cluster {clu.id}:\t{clu.size} documents')

    def compute_true_and_cluster_assignment(self, attrs):
        truth = {a.document_id: {a.aid} for a in attrs.itertuples() if pd.notnull(a.aid)}
        clu_map = {d: {c.id} for c in self.clusters for d in c.docs_ids if
                   d in [a.document_id for a in attrs.itertuples() if pd.notnull(a.aid)]}
        return truth, clu_map

    @staticmethod
    def compute_bcubed_metrics(ldict, cdict, verbose=False):
        """ See https://github.com/hhromic/python-bcubed """
        precision = bcubed.precision(cdict, ldict)
        recall = bcubed.recall(cdict, ldict)
        fscore = bcubed.fscore(precision, recall)

        if verbose:
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'Fscore: {fscore}')

        return {'b3prec': precision, 'b3recall': recall, 'b3Fscore': fscore}
