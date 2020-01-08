from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent


class Extractor(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract_features(self, data, args = None):
        """
        Extract features
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        pass

    @abstractmethod
    def select_features(self, data, args=None):
        """
        Select features
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        pass

    @abstractmethod
    def extract_select_features(self, data, args=None):
        """
        Extract-Select features
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """
        pass