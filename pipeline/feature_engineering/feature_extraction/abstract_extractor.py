from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent


class Extractor(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract_features(self, data, args = None):
        """
        Extract features
        :param data:
        :return: pandas.DataFrame
        """
        pass