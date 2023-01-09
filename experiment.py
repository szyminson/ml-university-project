#!/usr/bin/env python3
"""Perform an experiment"""

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.metrics import accuracy_score

random_state = 2023
n_splits = 5
n_repeats = 2

datasets = [
    # 'abalone-19_vs_10-11-12-13',
    'ecoli-0-1-4-6_vs_5',
    'ecoli-0-3-4_vs_5',
    'ecoli-0_vs_1',
    'ecoli4',
    # 'glass-0-1-5_vs_2',
    # 'glass0',
    # 'glass1',
    # 'haberman',
    # 'iris0',
    'pima',
    # 'poker-9_vs_7',
    # 'vehicle1',
    'vehicle2',
    'vowel0',
    # 'winequality-red-4',
    # 'winequality-white-3_vs_7',
    'wisconsin',
    'yeast1',
    'yeast-2_vs_4',
]

clfs = {
    'GNB': GaussianNB(),
    'CART': DecisionTreeClassifier(random_state=random_state),
    'SVM': SVC(random_state=random_state)
}

preprocs = {
    'none': None,
    'ros': RandomOverSampler(random_state=random_state),
    'smote': SMOTE(random_state=random_state),
    'bsmote': BorderlineSMOTE(random_state=random_state),
    'adasyn': ADASYN(random_state=random_state),
}
metrics = {
    'recall': recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
    'accuracy': accuracy_score,
}

def main():
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    metric_scores = np.zeros((len(clfs), len(datasets), len(
        preprocs), n_splits * n_repeats, len(metrics)))

    for clf_id, clf_key in enumerate(clfs):
        clf = clfs[clf_key]
        print('Classifier: ' + clf_key)
        for dataset_id, dataset_name in enumerate(datasets):
            print('\t'+dataset_name)
            dataset = np.genfromtxt("./datasets/%s.csv" %
                                    (dataset_name), delimiter=",")
            X = dataset[:, :-1]
            y = dataset[:, -1].astype(int)

            for fold_id, (train, test) in enumerate(rskf.split(X, y)):
                for preproc_id, preproc in enumerate(preprocs):
                    clf = clone(clf)

                    if preprocs[preproc] == None:
                        X_train, y_train = X[train], y[train]
                    else:
                        X_train, y_train = preprocs[preproc].fit_resample(
                            X[train], y[train])

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X[test])

                    for metric_id, metric in enumerate(metrics):
                        metric_scores[clf_id, dataset_id, preproc_id, fold_id, metric_id] = metrics[metric](
                            y[test], y_pred)

    np.save('results/results', metric_scores)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()

