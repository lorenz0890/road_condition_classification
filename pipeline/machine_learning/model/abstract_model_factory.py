from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent

class ModelFactory(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    def create_model(self, model_type, X, y, model_params, search_params):
        """
        :param X:
        :param y:
        :param model_params:
        :param selection_params:
        :param model_type:
        :return:
        """

    def find_optimal_model(self, mode, config, motif_list = None):
        """
        :return:
        """
        best_clf = None
        best_score = 0.0
        best_conf = None


