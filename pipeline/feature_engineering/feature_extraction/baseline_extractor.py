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
        print(args[0])
        print(args[1])
        print(args[2])
        X = extract_features(data, column_id = args[0], n_jobs = args[1], chunksize=args[2])
        X = impute(X)
        return X

    @overrides
    def select_features(self, data, args=None):
        """
        Select features
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        y = args[0]
        X_selected = select_features(data, y,
                                     ml_task='classification', n_jobs=args[1],
                                     chunksize=args[2], fdr_level=args[3])

        return X_selected

    @overrides
    def extract_select_training_features(self, data, args=None):
        """
        Extract-Select features
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """
        #print(args[0])
        #print(args[1])
        #print(args[2])
        #print(args[3])

        X = extract_features(data, column_id=args[0], n_jobs=args[1], chunksize=args[2])
        X = impute(X)

        y = args[3]
        X_selected = select_features(X, y,
                                     ml_task='classification', n_jobs=args[1],
                                     chunksize=args[2], fdr_level=args[4])

        return X_selected

    @overrides
    def extract_select_inference_features(self, data, args=None):
        """
        Extract-Select features
        Only extract specific features passed via args, we want the same as in taining
        https://stackoverflow.com/questions/50426458/retrieve-specific-features-by-using-tsfresh-in-python
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """
        X = extract_features(data, column_id=args[0], n_jobs=args[1], chunksize=args[2], kind_to_fc_parameters=args[3])
        X = impute(X)
        return X
