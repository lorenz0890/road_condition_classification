from pipeline.feature_engineering.feature_extraction.abstract_extractor import Extractor
from overrides import overrides
import pandas

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

        #min max var mean sum std aus min feature set von tshfresh
        X = pandas.DataFrame()
        segment_length = args[0]
        X['mean'] = data['acceleration_abs'].groupby(data.index // segment_length).mean()
        X['std'] = data['acceleration_abs'].groupby(data.index // segment_length).std()
        X['var'] = data['acceleration_abs'].groupby(data.index // segment_length).std()**(1/2)
        X['max'] = data['acceleration_abs'].groupby(data.index // segment_length).max()

        return X

    @overrides
    def select_features(self, data, args=None):
        """
        Select features
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        y_train = args[0]
        segment_length = args[1]
        y_train = data[['road_label', 'id']].reset_index(drop=True)
        y_train = y_train.groupby(y_train.index // segment_length).agg(lambda x: x.value_counts().index[0])
        data['id'] = y_train['id']
        data = data
        return data

    @overrides
    def extract_select_training_features(self, data, args=None):
        """
        Extract-Select features
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """
        X = pandas.DataFrame()
        segment_length = args[0]

        X['mean'] = data['acceleration_abs'].groupby(data.index // segment_length).mean()
        X['std'] = data['acceleration_abs'].groupby(data.index // segment_length).std()
        X['var'] = data['acceleration_abs'].groupby(data.index // segment_length).std() ** (1 / 2)
        X['max'] = data['acceleration_abs'].groupby(data.index // segment_length).max()

        y = data[['road_label', 'id']].reset_index(drop=True)
        y = y.groupby(y.index // segment_length).agg(lambda x: x.value_counts().index[0])
        X['id'] = y['id']
        X = X
        return X, y


    @overrides
    def extract_select_inference_features(self, data, args=None):
        """
        Extract-Select features
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """
        X = pandas.DataFrame()
        segment_length = args[0]

        X['mean'] = data['acceleration_abs'].groupby(data.index // segment_length).mean()
        X['std'] = data['acceleration_abs'].groupby(data.index // segment_length).std()
        X['var'] = data['acceleration_abs'].groupby(data.index // segment_length).std() ** (1 / 2)
        X['max'] = data['acceleration_abs'].groupby(data.index // segment_length).max()