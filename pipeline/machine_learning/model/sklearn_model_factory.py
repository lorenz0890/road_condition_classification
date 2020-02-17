from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV
import pickle
import pandas
import os
import traceback
from pipeline.machine_learning.model.abstract_model_factory import ModelFactory
from overrides import overrides
from scipy.stats import randint as sp_randint
from sklearn.metrics import confusion_matrix
import numpy

class SklearnModelFactory(ModelFactory):

    def __init__(self):
        super().__init__()


    @overrides
    def pre_clustering(self, X, y, args):
        """
        Preclustering and outlier detection using random forrests
        TODO: Make configureable
        :param X:
        :param y:
        :param args:
        :return:
        """
        y_clustering = IsolationForest(behaviour='new',
                                       max_samples='auto',
                                       n_jobs=-1,
                                       contamination=0.05,
                                       max_features=1.0,
                                       n_estimators=750
                                       ).fit_predict(X)

        X_combined = X.loc[pandas.DataFrame(y_clustering)[0] == 1]
        y_combined = y.loc[pandas.DataFrame(y_clustering)[0] == 1]

        X_combined = X_combined.reset_index(drop=True)
        y_combined = y_combined.reset_index(drop=True)
        return X_combined, y_combined

    @overrides
    def create_model(self, model_type, X_train, y_train, X_test, y_test, model_params, search_params):
        """
        Executes random search hyper parameter optimization for the specified model. Refer to sklearn
        documentation for details.
        Sources:
        # https://www.kaggle.com/hatone/mlpclassifier-with-gridsearchcv
        # https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        # alternative to grid search: https://github.com/sahilm89/lhsmdu
        :param model_type:
        :param X_train:
        :param y:
        :param model_params:
        :param search_params:
        :param test_size:
        :return:
        """
        try:

            if X_train is None or y_train is None or X_test is None or y_test is None or model_params is None or search_params is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if (not isinstance(X_train, pandas.DataFrame) and
                    not isinstance(X_train, pandas.core.frame.DataFrame) \
                    and not isinstance(X_train, pandas.core.series.Series)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if (not isinstance(y_train, pandas.DataFrame) and
                    not isinstance(y_train, pandas.core.frame.DataFrame) \
                    and not isinstance(y_train, pandas.core.series.Series)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if (not isinstance(X_test, pandas.DataFrame) and
                    not isinstance(X_test, pandas.core.frame.DataFrame) \
                    and not isinstance(X_test, pandas.core.series.Series)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if (not isinstance(y_test, pandas.DataFrame) and
                    not isinstance(y_test, pandas.core.frame.DataFrame) \
                    and not isinstance(y_test, pandas.core.series.Series)):
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

            return {'clf' : clf, 'X_test' : X_test, 'y_test' : y_test} #TODO returning X_test not necessary anymore, refacor

        except (TypeError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def find_optimal_model(self, mode, config, X_train_list=None, X_test_list=None):
        """
        :return:
        """
        best_clf = None
        best_score = 0.0
        best_conf = None
        best_X_train = None
        best_motif_radius = None
        best_motif_len = -1
        best_motif_count = -1

        for i in range(1, len(X_train_list)):
            X_train = X_train_list[i][0]
            y_train = X_train_list[i][1]
            X_test, y_test = None, None
            for j in range(1, len(X_test_list)):
                if X_train_list[i][2] == X_test_list[j][2] and X_train_list[i][3] == X_test_list[j][3]:
                    X_test = X_test_list[j][0]
                    y_test = X_test_list[j][1]

            print("------------------Iteration: {}-----------------".format(i))

            # Preclistering using iso forrests
            X_test, y_test = self.pre_clustering(X_test, y_test, None)
            X_train, y_train = self.pre_clustering(X_train, y_train, None)

            print('------------------Motifs-----------------')
            print("Motif radius: {}".format(X_train_list[i][2]))
            print("Motif length: {}".format(X_train_list[i][3]))
            print("Motif count: {}".format(X_train_list[i][4]))
            print("X shape: {}".format(X_train.shape))
            print("Training y label 1: {}".format(list(y_train[0]).count(1.0) / len(y_train))) #TODO: make configureable
            print("Training y label 3: {}".format(list(y_train[0]).count(3.0) / len(y_train)))
            print("Test y label 1: {}".format(list(y_test[0]).count(1.0) / len(y_test)))  # TODO: make configureable
            print("Test y label 3: {}\n\n".format(list(y_test[0]).count(3.0) / len(y_test)))
            if not (config['classifier_rep_class_distribution'][0] <
                    list(y_train[0]).count(1.0) / len(y_train) <
                    config['classifier_rep_class_distribution'][1]):
                print('Class distribution not representative in training set')
                continue
            if not (config['classifier_rep_class_distribution'][0] <
                    list(y_test[0]).count(1.0) / len(y_test) <
                    config['classifier_rep_class_distribution'][1]):
                print('Class distribution not representative in test set')
                continue

            # Test different classifiers on the detected features
            if (('sklearn_svc' in config['classifier_optimal_search_space'] or 'all' in config['classifier_optimal_search_space']) and
                X_train.shape[0] >= config['classifier_hypermaram_space_sklearn_svc']['cross_validation_k'] and
                X_test.shape[0] >= config['classifier_hypermaram_space_sklearn_rf']['cross_validation_k']
            ):

                print('------------------Sklearn-----------------')
                model = self.create_model(
                    model_type='svc',
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model_params={
                        'kernel': ['rbf', 'linear', 'poly'], #TODO use value from config
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
                                   config['classifier_hypermaram_space_sklearn_svc']['test_set_sz']] #TODO: This is deprecated at the classifier level. Remove from config and here.
                )
                print('------------------SVC-----------------')
                print(model['clf'].best_params_)
                score = model['clf'].score(X_test, y_test)
                y_pred = model['clf'].predict(X_test)
                conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
                if score > best_score:
                    best_clf = model['clf']
                    best_score = score
                    best_conf = conf
                    best_X_train = X_train
                    best_motif_radius = X_train_list[i][2]
                    best_motif_len = X_train_list[i][3]
                    best_motif_count = X_train_list[i][4]
                print(score)
                print(conf)
                print("\n\n")

            if (('sklearn_cart' in config['classifier_optimal_search_space'] or 'all' in config['classifier_optimal_search_space']) and
                    X_train.shape[0] >= config['classifier_hypermaram_space_sklearn_cart']['cross_validation_k'] and
                    X_test.shape[0] >= config['classifier_hypermaram_space_sklearn_rf']['cross_validation_k']
            ):

                model = self.create_model(
                    model_type='cart_tree',
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
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
                score = model['clf'].score(X_test, y_test)
                y_pred = model['clf'].predict(X_test)
                conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
                if score > best_score:
                    best_clf = model['clf']
                    best_score = score
                    best_conf = conf
                    best_X_train = X_train
                    best_motif_radius = X_train_list[i][2]
                    best_motif_len = X_train_list[i][3]
                    best_motif_count = X_train_list[i][4]
                print(score)
                print(conf)
                print("\n\n")

            if (('sklearn_rf' in config['classifier_optimal_search_space'] or 'all' in config['classifier_optimal_search_space']) and
                    X_train.shape[0] >= config['classifier_hypermaram_space_sklearn_rf']['cross_validation_k'] and
                    X_test.shape[0] >= config['classifier_hypermaram_space_sklearn_rf']['cross_validation_k']
            ):
                model = self.create_model(
                    model_type='random_forrest',
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
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
                score = model['clf'].score(X_test, y_test)
                y_pred = model['clf'].predict(X_test)
                conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
                if score > best_score:
                    best_clf = model['clf']
                    best_score = score
                    best_conf = conf
                    best_X_train = X_train
                    best_motif_radius = X_train_list[i][2]
                    best_motif_len = X_train_list[i][3]
                    best_motif_count = X_train_list[i][4]
                print(score)
                print(conf)
                print("\n\n")

            if (('sklearn_mlp' in config['classifier_optimal_search_space'] or 'all' in config['classifier_optimal_search_space']) and
                    X_train.shape[0] >= config['classifier_hypermaram_space_sklearn_mlp']['cross_validation_k'] and
                    X_test.shape[0] >= config['classifier_hypermaram_space_sklearn_rf']['cross_validation_k']
            ):
                architectures = []
                for architecture in config['classifier_hypermaram_space_sklearn_mlp']['architectures']:
                    architectures.append(tuple(architecture))

                model = self.create_model(
                    model_type='mlp_classifier',
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model_params={
                        'solver': config['classifier_hypermaram_space_sklearn_mlp']['solver'],
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
                print('------------------MLP----------------')
                print(model['clf'].best_params_)
                score = model['clf'].score(X_test, y_test)
                y_pred = model['clf'].predict(X_test)
                conf = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
                if score > best_score:
                    best_clf = model['clf']
                    best_score = score
                    best_conf = conf
                    best_X_train = X_train
                    best_motif_radius = X_train_list[i][2]
                    best_motif_len = X_train_list[i][3]
                    best_motif_count = X_train_list[i][4]
                print(score)
                print(conf)
                print("\n\n")



        print("best len", best_motif_len, "best radius", best_motif_radius)
        return best_clf, best_score, best_conf, best_X_train, best_motif_len, best_motif_radius, best_motif_count