from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent


class Preprocessor(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def segment_data(self, data, mode, label_column=None, args=None):
        """
        Reformat data
        :param data: pandas.DataFrame
        :param mode: 'string, 'semantic' for semantic segmentation, 'labels' for segementation per label
        :return: pandas.DataFrame(s)
        """
        pass

    @abstractmethod
    def de_segment_data(self, data_segments, selected_columns=None, axis = 0):
        """
        Reformat data
        :param data: pandas.DataFrame
        :param mode: 'string, 'semantic' for semantic segmentation, 'labels' for segementation per label
        :return: pandas.DataFrame(s)
        """
        pass

    @abstractmethod
    def remove_nans(self, data, replacement_mode, replacement_value = None):
        """
        Remove NaNs
        :param data: pandas.DataFrame
        :param replacement_mode: string, 'mean', 'replacement_val', 'delet_row'
        :param replacement_value: any type, used as value if replacment_mode is 'default_val'
        :return: pandas.DataFrame
        """
        pass


    @abstractmethod
    def remove_outliers_from_quantitative_data(self, data, replacement_mode, columns, quantile = None, threshold = None):
        """
        Remove statistical outliers,
        :param data: pandas.DataFrame
        :param replacement_mode: string, 'quantile' or 'threshold'
        :return: pandas.DataFrame
        :param quantile: float
        :param threshold: float
        """
        pass

    @abstractmethod
    def resample_quantitative_data(self, data, freq,  mode = None):
        """
        Apply a low/high pass filter to the dataset
        :param data: pandas.DataFrame
        :param freq: str
        :return: pandas.DataFrame
        """

    @abstractmethod
    def convert_unix_to_datetime(self, data, column, unit):
        """
        Convert unix time to date time
        :param data: pandas.DataFrame
        :param unit: string
        :param column: string
        :return:  pandas.DataFrame
        """
        pass

    @abstractmethod
    def remove_unwanted_labels(self, data, unwanted_labels, replacement_mode):
        """
        Remove unwanted data
        :param data: pandas.DataFrame
        :param unwanted_labels: list
        :param replacement_mode: str
        :return: pandas.DataFrame
        """
        pass

    @abstractmethod
    def label_data(self, labels, data):
        """
        Combine labels and data rows
        :param data: pandas.DataFrame
        :param labels: pandas.DataFrame
        :return: pandas.DataFrame
        """
        pass

    @abstractmethod
    def project_accelerometer_to_global_coordinates(self, data, target_columns, mode, args = None):
        """
        Project accelerometer data to a global coordinate system.
        This can be done, if gravity readings are known, by substracting them from the accelerometer readings,
        otherwise we can guess the direction of gravity as follows:

        "the mean of ac-celerometer readings along each axis over a period of time is a good estimate of the gravity
        direction"

        - Mizell. Using gravity to estimate accelerometer orientation.
        In ISWC ’03:Proceedings of the 7th IEEE International Symposium on Wearable Computers,page 252, Washington, DC, USA, 2003. IEEE Computer Society

        :param data: pandas.DataFrame
        :param target_columns: list, the columns containing accelerometer readings
        :param mode: string, 'mean_estimate_gravity', 'known_gravity', 'known_gyroscope', 'known_orientation'
        :param args: list , the columns containing, for example, gravity readings, but can be
        :return: pandas.DataFrame
        """
        pass

    @abstractmethod
    def znormalize_quantitative_data(self, data, columns = None):
        """
        Z-Normalize quantitative data
        Source:
            https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
        :param data: pandas.DataFrame
        :param columns: list, columns to be modified
        :return: pandas.DataFrame
        """
        pass

    def min_max_normalize_quantitative_data(self, data, columns=None):
        """
        Min max normalize quantitative data
        Source:
            https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
        :param data: pandas.DataFrame
        :param columns: columns to be modified
        :return: pandas.DataFrame
        """
        pass


    @abstractmethod
    def re_represent_data(self, current_representation, target_representation, data):
        """
        Apply uniform scaling to quantitative data
        :param data: pandas.DataFrame
        :param current_representation: string, 'SAX' 'DTW', 'numerical'
        :param target_representation: string, 'SAX' 'DTW', 'numerical'
        :return: pandas.DataFrame
        """
        pass

    @abstractmethod
    def reduce_quantitativ_data_dimensionality(self, data, mode, reduced_column_name = 'reduced', columns = None):
        """
        Reduce accelerometer dimensionality from 3 to 1 using L2 norm, PCA
        :param data: pandas.DataFrame
        :param mode: string, 'euclidean', 'manhatten'
        :param columns: list
        :return: pandas.DataFrame
        """
        pass

    @abstractmethod
    def encode_categorical_features(self, data, mode, columns, encoding_function):
        """
        :param data:
        :param mode:
        :param columns:
        :param encoding_function:
        :return:
        """
        pass
