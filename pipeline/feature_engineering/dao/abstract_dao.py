from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent


class DAO(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def read_data(self, file_path, column_names = None, use_rows = None, use_columns = None):
        """
        Load a dataset from disk. start_row and end_row can be used in case the data has to be loaded chunk-wise.
        :param file_path: string, path to data file
        :param column_names: array[string]
        :param use_rows: (int, int), (start index, end_index)
        :param use_columns: array[], column names
        :return: pandas.DataFrame, labeled and loaded data
        """
        pass

    @abstractmethod
    def write_features(self, file_path, data_dict):
        """
        Write extracted features to disk
        :param file_path:
        :param data_dict:
        :return:
        """
        pass

    @abstractmethod
    def load_features(self, file_path):
        """
        Load extracted features from disk
        :param file_path:
        :return:
        """
        pass

    @abstractmethod
    def write_model(self, file_path, model):
        """
        Write trained features to disk
        :param file_path:
        :param model:
        :return:
        """
        pass

    @abstractmethod
    def load_model(self, file_path):
        """
        Load trained model from disk
        :param file_path:
        :return:
        """
        pass