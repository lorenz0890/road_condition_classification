from pipeline.feature_engineering.feature_extraction.abstract_extractor import Extractor
from overrides import overrides
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

class BaselineExtractor(Extractor):

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
        X = extract_features(data, column_id = args[0], n_jobs = args[2], chunksize=args[3])
        X = impute(X)
        return X

    @overrides
    def select_features(self, data, args=None):
        """
        Select features
        :param data:
        :return: pandas.DataFrame
        """
        y = args[1]
        X_filtered = select_features(data, y,
                                     ml_task='classification', n_jobs=args[2],
                                     chunksize=args[3], fdr_level=args[4])

        return X_filtered