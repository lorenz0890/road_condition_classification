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
    def find_optimal_model(self, mode, config, motif_list=None):
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
            if int(len(X_train) * 0.2) < 50: #TODO Make configureable
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
            print("y label 1: {}".format(list(y_train[0]).count(1.0) / len(y_train))) #TODO: make configureable
            print("y label 3: {}\n\n".format(list(y_train[0]).count(3.0) / len(y_train)))

            # Test SVC on motif discovery
            if 'sklearn_svc' in config['classifier_optimal_search_space'] or 'all' in config['classifier_optimal_search_space']:
                print('------------------Sklearn-----------------')
                model = self.create_model(
                    model_type='svc',
                    X=X_train,
                    y=y_train,
                    model_params={
                        'kernel': ['rbf', 'linear', 'poly'],
                        'degree': sp_randint(config['classifier_hypermaram_space_sklearn_svc']['degree'][0],
                                             config['classifier_hypermaram_space_sklearn_svc']['degree'][1]),
                        'gamma': numpy.concatenate((10.0 ** -numpy.arange(0, config['classifier_hypermaram_space_sklearn_svc']['gamma_exponent']),
                                                    10.0 ** numpy.arange(1, config['classifier_hypermaram_space_sklearn_svc']['gamma_exponent']))),
                        'C': sp_randint(config['classifier_hypermaram_space_sklearn_svc']['C'][0],
                                        config['classifier_hypermaram_space_sklearn_svc']['C'][1]),
                        'max_iter': sp_randint(config['classifier_hypermaram_space_sklearn_svc']['max_iter'][0],
                                               config['classifier_hypermaram_space_sklearn_svc']['max_iter'][1]),
                        'shrinking': config['classifier_hypermaram_space_sklearn_svc']['shrinking'],
                        'probability': config['classifier_hypermaram_space_sklearn_svc']['probability'],
                        'random_state': sp_randint(config['classifier_hypermaram_space_sklearn_svc']['random_state'][0],
                                                   config['classifier_hypermaram_space_sklearn_svc']['random_state'][1]),
                    },
                    search_params=[config['hw_num_processors'],
                                   config['classifier_hypermaram_space_sklearn_svc']['verbose'],
                                   config['classifier_hypermaram_space_sklearn_svc']['cross_validation_k'],
                                   config['classifier_hypermaram_space_sklearn_svc']['iterations'],
                                   config['classifier_hypermaram_space_sklearn_svc']['save_classifier'],
                                   config['classifier_hypermaram_space_sklearn_svc']['save_classifier_file_name'],
                                   config['classifier_hypermaram_space_sklearn_svc']['test_set_sz']]
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

            if 'sklearn_cart' in config['classifier_optimal_search_space'] or 'all' in config['classifier_optimal_search_space']:
                model = self.create_model(
                    model_type='cart_tree',
                    X=X_train,
                    y=y_train,
                    model_params={
                        "max_depth": sp_randint(config['classifier_hypermaram_space_sklearn_cart']['max_depth'][0],
                                                config['classifier_hypermaram_space_sklearn_cart']['max_depth'][1]),
                        "max_features": sp_randint(1, X_train.shape[1]),
                        "min_samples_leaf": sp_randint(1, X_train.shape[1]),
                        "criterion": config['classifier_hypermaram_space_sklearn_cart']['criterion'],
                        'random_state': sp_randint(config['classifier_hypermaram_space_sklearn_cart']['random_state'][0],
                                                   config['classifier_hypermaram_space_sklearn_cart']['random_state'][1]),
                        'splitter': config['classifier_hypermaram_space_sklearn_cart']['splitter'],
                        'min_samples_split': sp_randint(config['classifier_hypermaram_space_sklearn_cart']['min_samples_split'][0],
                                                        config['classifier_hypermaram_space_sklearn_cart']['min_samples_split'][1])
                    },
                    search_params=[config['hw_num_processors'],
                                   config['classifier_hypermaram_space_sklearn_cart']['verbose'],
                                   config['classifier_hypermaram_space_sklearn_cart']['cross_validation_k'],
                                   config['classifier_hypermaram_space_sklearn_cart']['iterations'],
                                   config['classifier_hypermaram_space_sklearn_cart']['save_classifier'],
                                   config['classifier_hypermaram_space_sklearn_cart']['save_classifier_file_name'],
                                   config['classifier_hypermaram_space_sklearn_cart']['test_set_sz']]
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

            if 'sklearn_rf' in config['classifier_optimal_search_space'] or 'all' in config['classifier_optimal_search_space']:
                model = self.create_model(
                    model_type='random_forrest',
                    X=X_train,
                    y=y_train,
                    model_params={
                        'n_estimators': sp_randint(config['classifier_hypermaram_space_sklearn_rf']['n_estimators'][0],
                                                   config['classifier_hypermaram_space_sklearn_rf']['n_estimators'][1]),
                        'max_depth': sp_randint(config['classifier_hypermaram_space_sklearn_rf']['max_depth'][0],
                                                config['classifier_hypermaram_space_sklearn_rf']['max_depth'][1]),
                        'bootstrap': config['classifier_hypermaram_space_sklearn_rf']['bootstrap'],
                        'criterion': config['classifier_hypermaram_space_sklearn_rf']['criterion'],
                        'random_state': sp_randint(config['classifier_hypermaram_space_sklearn_rf']['random_state'][0],
                                                   config['classifier_hypermaram_space_sklearn_rf']['random_state'][1]),
                        'min_samples_split': sp_randint(config['classifier_hypermaram_space_sklearn_rf']['min_samples_split'][0],
                                                        config['classifier_hypermaram_space_sklearn_rf']['min_samples_split'][1])
                    },
                    search_params=[config['hw_num_processors'],
                                   config['classifier_hypermaram_space_sklearn_rf']['verbose'],
                                   config['classifier_hypermaram_space_sklearn_rf']['cross_validation_k'],
                                   config['classifier_hypermaram_space_sklearn_rf']['iterations'],
                                   config['classifier_hypermaram_space_sklearn_rf']['save_classifier'],
                                   config['classifier_hypermaram_space_sklearn_rf']['save_classifier_file_name'],
                                   config['classifier_hypermaram_space_sklearn_rf']['test_set_sz']]
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

            if 'sklearn_mlp' in config['classifier_optimal_search_space'] or 'all' in config['classifier_optimal_search_space']:
                architectures = []
                for architecture in config['classifier_hypermaram_space_sklearn_mlp']['architectures']:
                    architectures.append(tuple(architecture))

                model = self.create_model(
                    model_type='mlp_classifier',
                    X=X_train,
                    y=y_train,
                    model_params={
                        'solver': config['classifier_hypermaram_space_sklearn_mlp']['verbose'],
                        'max_iter': sp_randint(config['classifier_hypermaram_space_sklearn_mlp']['max_iter'][0],
                                               config['classifier_hypermaram_space_sklearn_mlp']['max_iter'][1]),
                        'alpha': numpy.concatenate((10.0 ** -numpy.arange(0, config['classifier_hypermaram_space_sklearn_mlp']['alpha_exponent']),
                                                    10.0 ** numpy.arange(1, config['classifier_hypermaram_space_sklearn_mlp']['alpha_exponent']))),
                        'hidden_layer_sizes': architectures,
                        'random_state': sp_randint(config['classifier_hypermaram_space_sklearn_mlp']['random_state'][0],
                                                   config['classifier_hypermaram_space_sklearn_mlp']['random_state'][1]),
                        'activation': config['classifier_hypermaram_space_sklearn_mlp']["activation_function"],
                        'learning_rate': config['classifier_hypermaram_space_sklearn_mlp']["learning_rate"],
                        'learning_rate_init': numpy.concatenate(
                            (10.0 ** -numpy.arange(0, config['classifier_hypermaram_space_sklearn_mlp']["learning_rate_init_exponent"]),
                             10.0 ** numpy.arange(1, config['classifier_hypermaram_space_sklearn_mlp']["learning_rate_init_exponent"]))),
                        'batch_size': sp_randint(config['classifier_hypermaram_space_sklearn_mlp']['batch_size'][0],
                                                   config['classifier_hypermaram_space_sklearn_mlp']['batch_size'][1]),
                        'shuffle': config['classifier_hypermaram_space_sklearn_mlp']['shuffle'],
                        'early_stopping': config['classifier_hypermaram_space_sklearn_mlp']['early_stopping'],
                    },
                    search_params=[config['hw_num_processors'],
                                   config['classifier_hypermaram_space_sklearn_mlp']['verbose'],
                                   config['classifier_hypermaram_space_sklearn_mlp']['cross_validation_k'],
                                   config['classifier_hypermaram_space_sklearn_mlp']['iterations'],
                                   config['classifier_hypermaram_space_sklearn_mlp']['save_classifier'],
                                   config['classifier_hypermaram_space_sklearn_mlp']['save_classifier_file_name'],
                                   config['classifier_hypermaram_space_sklearn_mlp']['test_set_sz']]
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