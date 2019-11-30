from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent


class ModelFactory(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _create_random_forrest(self, X, y, model_params, search_params, test_size):
        """
        :param X:
        :param y:
        :param model_types:
        :param hyperparms:
        :return:
        """
        pass

    @abstractmethod
    def _create_CART_tree(self, X, y, model_params, search_params, test_size):
        """
        :param X:
        :param y:
        :param model_types:
        :param hyperparms:
        :return:
        """
        pass

    @abstractmethod
    def _create_SVC(self, X, y, model_params, search_params, test_size):
        """
        :param X:
        :param y:
        :param model_types:
        :param hyperparms:
        :return:
        """
        pass

    @abstractmethod
    def _create_MLP_classifier(self, X, y, model_params, search_params, test_size):
        """
        :param X:
        :param y:
        :param model_types:
        :param hyperparms:
        :return:
        """
        pass

    def create_model(self, model_type, X, y, model_params, search_params, test_size):
        """
        :param X:
        :param y:
        :param model_params:
        :param selection_params:
        :param model_type:
        :return:
        """



