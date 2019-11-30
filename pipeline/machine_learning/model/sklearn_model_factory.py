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

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=search_params[6])
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

            return clf, X_test, y_test

        except (TypeError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)
