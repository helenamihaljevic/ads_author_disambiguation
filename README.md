# README

Project to disambiguate blocks of authorships in the SAO/NASA Astrophysics Data System (ADS) via a semi-supervised approach:

1. We have blocked all authorships that share the same surname and first initial after being pre-processed by removing diacritics and lowercasing. 
2. We have then trained a classification model to decide whether two articles within a given block belong to the same author. 
3. The trained classifier is used to create author entities as subgraphs of authorship graphs. These are constructed as follows: the authorships of the block are nodes of the graph; an edge is drawn between two nodes if the classifier predicts that both are authored by the same person. Additionally, we use the classifier's class probabilities as edge weights, which results in a labeled graph. 
4. Author profiles are constructed as communities through label propagation. This step can be replaced by any other graph clustering algorithm (that requires no further parameter specification or tuning), e.g. $k$-cliques.

For training and evaluation we have manually disambiguated the majority (84.4\%) of authorships from 137 authorship blocks containing at least 3 authorships and amounting to 14,054 records. The sample is constructed by  chosing 6 from the largest 10 blocks to account for particularly challenging cases with many homonyms; the remaining were randomly sampled. 

## Training pipeline

1. Set `MODEL_FEATURES` in `modelling_config.py` to those features that you want to use to train a classifier for pairwise comparison. 
Also select models and other configuration parameters in this file.
2. Run script `scripts/precompute_matrices.py` in order to have all feature matrices ready for training and test. This will take some time.
3. Run script `scripts/train_classifier.py` for cross validated training with hyperparameter optimization
4. If happy with results, set the right value for MODEL_TYPE in script `scripts/train_final_classifier.py`
5. Perform and evaluate clusering using `scripts/evaluate_clustering.py`.



