from pipeline.feature_engineering.feature_extraction.abstract_extractor import Extractor
from overrides import overrides

class ManualExtractor(Extractor):

    def __init__(self):
        super().__init__()

    @overrides
    def extract_features(self, data, args = None):
        """
        Extract features
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        pass

    @overrides
    def select_features(self, data, args=None):
        """
        Select features
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        return data

    @overrides
    def extract_select_training_features(self, data, args=None):
        """
        Extract-Select features
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """
        pass

    @overrides
    def extract_select_inference_features(self, data, args=None):
        """
        Extract-Select features
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """
        pass