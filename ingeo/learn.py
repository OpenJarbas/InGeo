#!/usr/bin/python3

import functools
import gzip
import logging
import math
import operator
import pickle
from random import shuffle
from threading import Thread

import numpy
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ingeo.fingerprint import LocationFingerprints


def _timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception(f'function [{func.__name__}] timeout [{timeout}] exceeded!')]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


class InGeo:
    clf_names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA"]

    def __init__(self, family):
        self.db = LocationFingerprints(family)
        self.family = family
        self.algorithms = {}
        self._results = []
        self._data = numpy.zeros(len(self.db["sensors"]))
        self.logger = self._init_logger()

    def _init_logger(self):
        # create logger with 'spam_application'
        logger = logging.getLogger(f'InGeo.learn.{self.family}')
        logger.setLevel(logging.DEBUG)
        # TODO xdg
        fh = logging.FileHandler(f'/tmp/InGeo.learn.{self.family}.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - [%(name)s/%(funcName)s] - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logging.getLogger(f'InGeo.Learn.{self.family}')

    def classify(self, sensor_data):
        # data 2 numpy
        data = numpy.zeros(len(self.db["sensors"]))
        for sensor_type in sensor_data:
            for sensor in sensor_data[sensor_type]:
                sensor_name = sensor_type + "-" + sensor
                if sensor_name in self.db["sensors"]:
                    data[self.db["sensors"][sensor_name]] = sensor_data[sensor_type][sensor]
        self._data = data.reshape(1, -1)

        # classify in threads
        threads = [None] * len(self.algorithms)
        self._results = [None] * len(self.algorithms)
        payload = {'location_names': {v: k for k, v in self.db["locations"].items()},
                   'predictions': []}

        for i, alg in enumerate(self.algorithms.keys()):
            threads[i] = Thread(target=self._do_classification, args=(i, alg))
            threads[i].start()

        for i, _ in enumerate(self.algorithms.keys()):
            threads[i].join()

        for result in self._results:
            if result is not None:
                payload['predictions'].append(result)
        return payload

    def predict(self, sensor_data):
        classified = self.classify(sensor_data)
        score = {}
        preds = [(p["locations"], p["probabilities"])
                 for p in classified["predictions"]]
        for loc, prob in preds:
            for l, p in zip(loc, prob):
                if l not in score:
                    score[l] = 0
                score[l] += p
        return {classified["location_names"][int(k)]: v / len(preds)
                for k, v in score.items()}

    def _do_classification(self, index, name):
        try:
            prediction = self.algorithms[name].predict_proba(self._data)
        except Exception as e:
            self.logger.error(self._data)
            self.logger.error(str(e))
            return
        predict = {}
        for i, pred in enumerate(prediction[0]):
            predict[i] = pred
        predict_payload = {'name': name,
                           'locations': [], 'probabilities': []}
        badValue = False
        for tup in sorted(predict.items(), key=operator.itemgetter(1), reverse=True):
            predict_payload['locations'].append(str(tup[0]))
            predict_payload['probabilities'].append(round(float(tup[1]), 2))
            if math.isnan(tup[1]):
                badValue = True
                break
        if badValue:
            return
        self._results[index] = predict_payload

    @_timeout(10)
    def _train(self, clf, x, y):
        return clf.fit(x, y)

    def learn(self):
        rows = []
        for location, prints in self.db["fingerprints"].items():
            rows += prints
        if not rows:
            raise ValueError(f"No training data! {self.db.path}")
        # first column in row is the classification, Y
        y = numpy.zeros(len(rows))
        x = numpy.zeros((len(rows), len(rows[0]) - 1))

        # shuffle it up for training
        record_range = list(range(len(rows)))
        shuffle(record_range)
        for i in record_range:
            y[i] = rows[i][0]
            x[i, :] = numpy.array(rows[i][1:])

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, probability=True),
            SVC(gamma=2, C=1, probability=True),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        self.algorithms = {}
        for name, clf in zip(self.clf_names, classifiers):
            self.logger.debug(f"learning {name}")
            try:
                self.algorithms[name] = self._train(clf, x, y)
                self.logger.debug(f"learned {name}")
            except Exception as e:
                self.logger.error(f"{name} {e}")

    def save(self, save_file=None):
        save_file = save_file or self.db.path.replace(".json", ".ingeo")
        f = gzip.open(save_file, 'wb')
        pickle.dump(self.family, f)
        pickle.dump(self.algorithms, f)
        f.close()
        self.db.store()

    def load(self, save_file=None):
        save_file = save_file or self.db.path.replace(".json", ".ingeo")
        f = gzip.open(save_file, 'rb')
        self.family = pickle.load(f)
        self.algorithms = pickle.load(f)
        f.close()
        self.db = LocationFingerprints(self.family)


if __name__ == "__main__":

    ingeo = InGeo("test")
    # train on test data
    ingeo.db.import_csv('../test/test.csv')
    ingeo.learn()

    # save models
    ingeo.save("/tmp/test.ingeo")
    ingeo.load("/tmp/test.ingeo")

    # predict
    data = {
        "wifi": {
            "a": -76.3,
            "yellowyellowyellow": -68.1
        }
    }

    print(ingeo.classify(data))
    # {'location_names': {0: 'artroom', 1: 'living', 2: 'kitchen', 3: 'desk'},
    # 'predictions': [
    # {'name': 'Nearest Neighbors', 'locations': ['3', '0', '1', '2'], 'probabilities': [1.0, 0.0, 0.0, 0.0]},
    # {'name': 'Linear SVM', 'locations': ['3', '0', '2', '1'], 'probabilities': [0.53, 0.33, 0.09, 0.06]},
    # {'name': 'RBF SVM', 'locations': ['3', '1', '0', '2'], 'probabilities': [0.94, 0.04, 0.01, 0.01]},
    # {'name': 'Decision Tree', 'locations': ['3', '1', '0', '2'], 'probabilities': [0.99, 0.01, 0.0, 0.0]},
    # {'name': 'Random Forest', 'locations': ['3', '0', '1', '2'], 'probabilities': [0.94, 0.03, 0.02, 0.01]},
    # {'name': 'Neural Net', 'locations': ['3', '0', '1', '2'], 'probabilities': [0.61, 0.29, 0.06, 0.04]},
    # {'name': 'AdaBoost', 'locations': ['3', '1', '0', '2'], 'probabilities': [0.68, 0.24, 0.08, 0.0]},
    # {'name': 'Naive Bayes', 'locations': ['0', '3', '1', '2'], 'probabilities': [0.4, 0.31, 0.22, 0.07]},
    # {'name': 'QDA', 'locations': ['0', '3', '1', '2'], 'probabilities': [0.4, 0.31, 0.21, 0.07]}]}

    print(ingeo.predict(data))
    # {'desk': 0.701111111111111,
    # 'artroom': 0.1711111111111111,
    # 'living': 0.09555555555555556,
    # 'kitchen': 0.03222222222222223}
