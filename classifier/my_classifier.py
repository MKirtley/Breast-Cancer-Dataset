import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from collections import Counter
import pprint

"""
Random Forest Classifier that uses SciKit-learn's Decision trees.
Understood and modified from python-engineer's example.
https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/random_forest.py
"""


class MyClassifier:

    def __init__(self, init_msg=None,
                 n_estimators=100,
                 random_state=1,
                 criterion="entropy",
                 max_depth=100,
                 min_samples_split=2):
        """
        Init function.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.init_msg = init_msg

        if init_msg:
            print(init_msg)

    def get_params(self, deep=True):
        """
        Get model parameters.
        Required by Scikit-Learn.
        :param deep: Not used
        :return: Dictionary of parameter names and values.
        """
        return {"init_msg": self.init_msg,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "criterion": self.criterion,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split}

    def set_params(self, **parameters):
        """
        Set model parameters.
        Required by Scikit-Learn.
        :param parameters: Dictionary of parameters and their values.
        :return: self.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def random_sample(self, X, y):
        """
        Gives each tree a random sample of instances.
        Each sample ranges from 0 to all instances.
        (Bootstrapping).
        :param X: Feature matrix
        :param y: labels (vector)
        :return: A random sample of instances for each tree.
        """
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        The training phase.
        Each tree is SciKit-Learn's Decision Tree classifier (CART).
        :param X: Feature matrix
        :param y: labels (vector)
        :return: self
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.trees = []

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(random_state=self.random_state,
                                          criterion=self.criterion,
                                          max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split)
            X_sample, y_sample = self.random_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def majority_vote(self, y):
        """
        Counts the predictions from each tree and returns
        the majority vote.
        If there's a tie, class as 1 (Malignant).
        :param y:
        :return: majority vote
        """
        try:
            votes = Counter(y)
            winner = votes.most_common(2)[0][0]
            first_pos = votes.most_common(2)[0][1]
            second_pos = votes.most_common(2)[1][1]

            if first_pos == second_pos:
                majority = 1
            else:
                majority = winner
        except IndexError:
            votes = Counter(y)
            majority = votes.most_common(2)[0][0]
        return majority

    def predict(self, X):
        """
        Making a prediction with each tree.
        :param X: feature matrix
        :return: list of classification labels
        """

        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)

        y_pred = [self.majority_vote(tree_pred) for tree_pred in predictions]

        return np.array(y_pred)

    def score(self, X, y):
        """
        Default scorer.
        :param X: Feature matrix
        :param y: Label vector
        :return: Returns accuracy as the score.
        """
        return metrics.accuracy_score(self.predict(X), y)


if __name__ == '__main__':
    my_classifier = MyClassifier()

    X_train = np.array([[10, 12, 11], [101, 121, 111], [102, 122, 112], [11, 13, 12]])
    y_train = np.array([0, 1, 2, 0])

    X_test = np.array([[10, 12, 11], [101, 121, 111], [102, 122, 112], [10, 12, 11], [101, 121, 111], [102, 122, 112]])

    my_classifier.fit(X_train, y_train)
    y_pred = my_classifier.predict(X_test)

    pprint.pprint(y_pred)
