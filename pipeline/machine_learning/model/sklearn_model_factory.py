from pipeline.machine_learning.model.abstract_model_factory import ModelFactory
from overrides import overrides
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import os
import traceback
import pandas

class SklearnModelFactory(ModelFactory):

    def __init__(self):
        super().__init__()

    @overrides
    def _create_random_forrest(self, X, y, model_params, selection_params):

        try:
            if X is None or y is None or model_params is None or selection_params is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if (not isinstance(X, pandas.DataFrame) and
                    not isinstance(X, pandas.core.frame.DataFrame) \
                    and not isinstance(X, pandas.core.series.Series)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if (not isinstance(y, pandas.DataFrame) and
                    not isinstance(y, pandas.core.frame.DataFrame) \
                    and not isinstance(y, pandas.core.series.Series)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if not isinstance(model_params, list) or not isinstance(selection_params, list):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            clf = RandomForestClassifier(n_estimators=model_params[0], max_depth=model_params[1], random_state=model_params[2])
            tscv = ShuffleSplit(n_splits=selection_params[0])#TimeSeriesSplit(n_splits=selection_params[0])
            evaluated_estimators = cross_validate(clf, X, y, cv=tscv, return_estimator=True)
            return evaluated_estimators

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)


    @overrides
    def _create_CART_tree(self, X, y, model_params, selection_params):
        clf = DecisionTreeClassifier(random_state=model_params[0])
        tscv = ShuffleSplit(n_splits=selection_params[0])# TimeSeriesSplit(n_splits=selection_params[0])
        evaluated_estimators = cross_validate(clf, X, y, cv=tscv, return_estimator=True)
        return evaluated_estimators

    @overrides
    def _create_SVC(self, X, y, model_params, selection_params):
        clf = SVC(gamma=model_params[0])
        evaluated_estimators = cross_validate(clf, X, y, cv=selection_params[0], return_estimator=True)
        return evaluated_estimators

    @overrides
    def _create_MLP_classifier(self, X, y, model_params, selection_params):
        clf = MLPClassifier(solver=model_params[0], alpha=model_params[1],
                            hidden_layer_sizes=model_params[2], random_state=model_params[3])

        tscv = ShuffleSplit(n_splits=selection_params[0])# TimeSeriesSplit(n_splits=selection_params[0])
        evaluated_estimators = cross_validate(clf, X, y, cv=tscv, return_estimator=True)
        return evaluated_estimators

    @overrides
    def create_model(self, model_type, X, y, model_params, selection_params):

        if model_type == 'random_forrest':
            return self._create_random_forrest(X, y, model_params, selection_params)

        if model_type == 'cart_tree':
            return self._create_CART_tree(X, y, model_params, selection_params)

        if model_type == 'svc':
            return self._create_SVC(X, y, model_params, selection_params)

        if model_type == 'mlp_classifier':
            return self._create_MLP_classifier(X, y, model_params, selection_params)

