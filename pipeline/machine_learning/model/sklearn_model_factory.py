def warn(*args, **kwargs):
    """
    Hack sklearn warnings away, temporary fix
    https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
    :param args:
    :param kwargs:
    :return:
    """
    pass
import warnings
warnings.warn = warn

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
import pandas
import os
import traceback
from pipeline.machine_learning.model.abstract_model_factory import ModelFactory
from overrides import overrides
from sklearn.model_selection import train_test_split
from scipy.stats import randint as sp_randint
from sklearn.metrics import confusion_matrix
import numpy


class SklearnModelFactory(ModelFactory):

    def __init__(self):
        super().__init__()


    @overrides
    def create_model(self, model_type, X, y, model_params, search_params):
        """
        Executes random search hyper parameter optimization for the specified model. Refer to sklearn
        documentation for details.
        Sources:
        # https://www.kaggle.com/hatone/mlpclassifier-with-gridsearchcv
        # https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        # alternative to grid search: https://github.com/sahilm89/lhsmdu
        :param model_type:
        :param X:
        :param y:
        :param model_params:
        :param search_params:
        :param test_size:
        :return:
        """
        try:

            if X is None or y is None or model_params is None or search_params is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if (not isinstance(X, pandas.DataFrame) and
                    not isinstance(X, pandas.core.frame.DataFrame) \
                    and not isinstance(X, pandas.core.series.Series)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if (not isinstance(y, pandas.DataFrame) and
                    not isinstance(y, pandas.core.frame.DataFrame) \
                    and not isinstance(y, pandas.core.series.Series)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if not isinstance(model_params, dict) or not isinstance(search_params, list):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            model = None
            if model_type == 'random_forrest':
                model = RandomForestClassifier()

            if model_type == 'cart_tree':
                model = DecisionTreeClassifier()

            if model_type == 'svc':
                model = SVC()

            if model_type == 'mlp_classifier':
                model = MLPClassifier()

            if model is None:
                raise ValueError(self.messages.PROVIDED_MODE_DOESNT_EXIST.value)

            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=search_params[6],
                                                                stratify=y
                                                                )

            clf = RandomizedSearchCV(model,
                                     model_params,
                                     n_jobs=search_params[0],
                                     verbose=search_params[1],
                                     cv=search_params[2],
                                     n_iter=search_params[3]
                                     )
            clf.fit(X_train, y_train)
            if search_params[4]:
                with open(r"{}".format(search_params[5]), "wb") as output_file:
                    pickle.dump(clf, output_file)

            return {'clf' : clf, 'X_test' : X_test, 'y_test' : y_test}

        except (TypeError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def find_optimal_model(self, mode, motif_list=None, model_list = ['all']):
        """
        :return:
        """
        best_clf = None
        best_score = 0.0
        best_conf = None
        best_X_train = None
        best_y_train = None
        best_motif_len = -1
        best_motif_count = -1

        for i in range(1, len(motif_list)):
            X_train = motif_list[i][0]  # [:3000]
            y_train = motif_list[i][1]  # [:3000]
            print("------------------Iteration: {}-----------------".format(i))
            if int(len(X_train) * 0.2) < 50:
                print('Test set too small')
                continue
            if not (0.35 < list(y_train[0]).count(1.0) / len(y_train) < 0.65): #TODO distribution boundaries shgould be configureable
                print('Class distribution not representative')
                continue

            print('------------------Motifs-----------------')
            print("Motif radius: {}".format(motif_list[i][2]))
            print("Motif length: {}".format(motif_list[i][3]))
            print("Motif count: {}".format(motif_list[i][4]))
            print("X shape: {}".format(X_train.shape))
            print("y label 1: {}".format(list(y_train[0]).count(1.0) / len(y_train)))
            print("y label 3: {}\n\n".format(list(y_train[0]).count(3.0) / len(y_train)))

            # Test SVC on motif discovery
            if 'sklearn_svc' in model_list or 'all' in model_list:
                print('------------------Sklearn-----------------')
                model = self.create_model(
                    model_type='svc',
                    X=X_train,
                    y=y_train,
                    model_params={
                        'kernel': ['rbf', 'linear', 'poly'],
                        'degree': sp_randint(2, X_train.shape[1] * 3),
                        'gamma': numpy.concatenate((10.0 ** -numpy.arange(0, 10), 10.0 ** numpy.arange(1, 10))),
                        'C': sp_randint(2, 5000),
                        'max_iter': sp_randint(2, 5000),
                        'shrinking': [True, False],
                        'probability': [True, False],
                        'random_state': sp_randint(1, 10),
                    },
                    search_params=[-1, 0, 10, 100, True, "svc_rs.pickle", 0.2]
                )
                print('------------------SVC-----------------')
                print(model['clf'].best_params_)
                X_test, y_test = model['X_test'], model['y_test']
                score = model['clf'].score(X_test, y_test)
                y_pred = model['clf'].predict(X_test)
                conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
                if score > best_score:
                    best_clf = model['clf']
                    best_score = score
                    best_conf = conf
                    best_X_train = X_train
                    #best_y_train = y_train
                    best_motif_len = motif_list[i][3]
                    best_motif_count = motif_list[i][4]
                print(score)
                print(conf)
                print("\n\n")

            if 'sklearn_cart' in model_list or 'all' in model_list:
                model = self.create_model(
                    model_type='cart_tree',
                    X=X_train,
                    y=y_train,
                    model_params={
                        "max_depth": sp_randint(1, 128),
                        "max_features": sp_randint(1, X_train.shape[1]),
                        "min_samples_leaf": sp_randint(1, X_train.shape[1]),
                        "criterion": ["gini", "entropy"],
                        'random_state': sp_randint(1, 10),
                        'splitter': ['best', 'random'],
                        'min_samples_split': sp_randint(2, 10)
                    },
                    search_params=[-1, 0, 10, 100, True, "dt_rs.pickle", 0.2]
                )
                print('------------------CART-Tree-----------------')
                print(model['clf'].best_params_)
                X_test, y_test = model['X_test'], model['y_test']
                score = model['clf'].score(X_test, y_test)
                y_pred = model['clf'].predict(X_test)
                conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
                if score > best_score:
                    best_clf = model['clf']
                    best_score = score
                    best_conf = conf
                    best_X_train = X_train
                    #best_y_train = y_train
                    best_motif_len = motif_list[i][3]
                    best_motif_count = motif_list[i][4]
                print(score)
                print(conf)
                print("\n\n")

            if 'sklearn_rf' in model_list or 'all' in model_list:
                model = self.create_model(
                    model_type='random_forrest',
                    X=X_train,
                    y=y_train,
                    model_params={
                        'n_estimators': sp_randint(1, 100),
                        'max_depth': sp_randint(1, 128),
                        # 'max_features': sp_randint(1, X_train.shape[1]),
                        'min_samples_split': sp_randint(2, X_train.shape[1]),
                        'bootstrap': [True, False],
                        "criterion": ["gini", "entropy"],
                        'random_state': sp_randint(1, 10),
                        #'min_samples_split': sp_randint(2, 10)
                    },
                    search_params=[-1, 0, 10, 100, True, "rf_rs.pickle", 0.2]
                )
                print('------------------Random Forrest----------------')
                print(model['clf'].best_params_)
                X_test, y_test = model['X_test'], model['y_test']
                score = model['clf'].score(X_test, y_test)
                y_pred = model['clf'].predict(X_test)
                conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
                if score > best_score:
                    best_clf = model['clf']
                    best_score = score
                    best_conf = conf
                    best_X_train = X_train
                    #best_y_train = y_train
                    best_motif_len = motif_list[i][3]
                    best_motif_count = motif_list[i][4]
                print(score)
                print(conf)
                print("\n\n")

            if 'sklearn_mlp' in model_list or 'all' in model_list:
                model = self.create_model(
                    model_type='mlp_classifier',
                    X=X_train,
                    y=y_train,
                    model_params={
                        'solver': ['adam', 'lbfgs', 'sgd'],
                        'max_iter': sp_randint(1, 250),
                        'alpha': numpy.concatenate((10.0 ** -numpy.arange(0, 10), 10.0 ** numpy.arange(1, 10))),
                        'hidden_layer_sizes': [  # (128,128,128,128), #architecture see
                            # (128,128,128),
                            (128, 128),
                            (128),
                            # (64,64,64,64),
                            # (64,64,64),
                            (64, 64),
                            (64),
                            # (32,32,32,32),
                            # (32,32,32),
                            (32, 32),
                            (32),
                            # (16,16,16,16),
                            # (16,16,16),
                            (16, 16),
                            (16)
                        ],
                        'random_state': sp_randint(1, 10),
                        'activation': ["logistic", "relu", "tanh"],
                        'learning_rate': ['constant', 'invscaling', 'adaptive'],
                        'learning_rate_init': numpy.concatenate(
                            (10.0 ** -numpy.arange(0, 10), 10.0 ** numpy.arange(1, 10))),
                        'batch_size': sp_randint(1, 10),
                        'shuffle': [True, False],
                        'early_stopping': [True, False],
                    },
                    search_params=[-1, 0, 10, 25, True, "mlp_rs.pickle", 0.2]
                )
                print(model['clf'].best_params_)
                X_test, y_test = model['X_test'], model['y_test']
                score = model['clf'].score(X_test, y_test)
                y_pred = model['clf'].predict(X_test)
                conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
                if score > best_score:
                    best_clf = model['clf']
                    best_score = score
                    best_conf = conf
                    best_X_train = X_train
                    #best_y_train = y_train
                    best_motif_len = motif_list[i][3]
                    best_motif_count = motif_list[i][4]
                print(score)
                print(conf)
                print("\n\n")

        return best_clf, best_score, best_conf, best_X_train, best_motif_len, best_motif_count