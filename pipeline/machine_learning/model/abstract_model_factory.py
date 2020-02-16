from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent

class ModelFactory(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    def create_model(self, model_type, X_train, y_train, X_test, y_test, model_params, search_params):
        """
        :param X:
        :param y:
        :param model_params:
        :param selection_params:
        :param model_type:
        :return:
        """

    def find_optimal_model(self, mode, config, X_train_list=None, X_test_list=None):
        """
        :return:
        """


