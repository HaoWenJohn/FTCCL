import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

def fit_svm(features, y, MAX_SAMPLES=10000):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=np.inf, gamma='scale')
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm, {
                'C': [
                    0.001, 0.001, 0.01, 0.1, 1, 10, 100,
                    np.inf
                ],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [30000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]
            
        grid_search.fit(features, y)
        return grid_search.best_estimator_

