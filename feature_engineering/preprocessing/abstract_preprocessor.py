from abc import ABC, abstractmethod
from utils.rctc_component import RCTCComponent

class Preprocessor(ABC, RCTCComponent):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def segment_data(self, data, mode, label_column=None, support=None):
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
    def remove_outliers_from_quantitative_data(self, data, threshold, replacement_mode):
        """
        Remove statistical outliers,
        :param data: pandas.DataFrame
        :param threshold:
        :param replacement_mode:
        :return:
        """
        pass

    @abstractmethod
    def remove_unwanted_labels(self, data, unwanted_labels, replacement_mode):
        """
        Remove unwanted data
        :param data: pandas.DataFrame
        :param unwanted_labels:
        :param replacement_mode:
        :return:
        """
        pass

    @abstractmethod
    def label_data(self, labels, data):
        """
        Combine labels and data rows
        :param data: pandas.DataFrame
        :return: pandas.DataFrame
        """
        pass

    @abstractmethod
    def project_accelerometer_to_global_coordinates(self, data, target_columns, mode, support_columns = None):
        """
        Project accelerometer data to a global coordinate system.
        This can be done, if gravity readings are known, by substracting them from the accelerometer readings,
        otherwise we can guess the direction of gravity as follows:

        "the mean of ac-celerometer readings along each axis over a period of time is a good estimate of the gravity
        direction"

        - Mizell. Using gravity to estimate accelerometer orientation.
        In ISWC ’03:Proceedings of the 7th IEEE International Symposium on Wearable Computers,page 252, Washington, DC, USA, 2003. IEEE Computer Society

        :param data: pandas.DataFrame
        :param target_columns: array['string], the columns containing accelerometer readings
        :param mode: string, 'mean_estimate_gravity', 'known_gravity', 'known_gyroscope', 'known_orientation'
        :param support_columns: array['string] , the columns containing, for example, gravity readings
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
        :param columns: columns to be modified
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
    def norm_quantitative_data(self, norm, represenation, data, columns = None):
        """
        Apply norm to quantitative data
        :param data: pandas.DataFrame
        :param norm: string, 'euclidean', 'manhatten'
        :param represenation: string, 'SAX' 'DTW', 'numerical'
        :return: pandas.DataFrame
        """
        pass