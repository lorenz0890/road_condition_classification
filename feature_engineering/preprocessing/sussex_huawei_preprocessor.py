from feature_engineering.preprocessing.abstract_preprocessor import Preprocessor
from feature_engineering.preprocessing.replacement_strategies.mean_replacement_strategy import MeanReplacementStrategy
from feature_engineering.preprocessing.replacement_strategies.del_row_replacement_strategy import DelRowReplacementStrategy
from feature_engineering.preprocessing.replacement_strategies.replacement_val_replacement_strategy import ReplacementValReplacementStrategy
from overrides import overrides
import traceback
import os
import pandas

class SussexHuaweiPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__()

    @overrides
    def segment_data(self, data, mode, label_column=None, support=None):
        if data is None or mode is None:
            raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
        if not isinstance(data, pandas.DataFrame):
            raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

        if mode == 'semantic':
            raise NotImplementedError

        if mode == 'labels':
            # 1. Select all data with desired label value
            data_segments = []
            for target_label in support:
                selected_data = data[data[label_column] == target_label]

                # 2. Split by non-subsequent indices
                # Source for next 3 lines after comment:
                # https://stackoverflow.com/questions/56257329/how-to-split-a-dataframe-based-on-consecutive-index
                non_sequence = pandas.Series(selected_data.index).diff() != 1
                grouper = non_sequence.cumsum().values
                selected_data_segments = [group for _, group in selected_data.groupby(grouper)]

                for segment in selected_data_segments:
                    data_segments.append(segment)
            return data_segments



    @overrides
    def remove_nans(self, data, replacement_mode, replacement_value=None):
        """
        Remove NaNs
        :param data:
        :param replacement_mode: string, 'mean', 'replacement_val', 'delet_row'
        :param replacement_value: any type, used as value if replacment_mode is 'default_val'
        :return: pandas.DataFrame
        """
        try:
            if data is None or replacement_mode is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if replacement_mode == 'mean':
                return MeanReplacementStrategy().replace(data, 'NaN')
            if replacement_mode == 'del_row':
                return DelRowReplacementStrategy().replace(data, 'NaN')
            if replacement_mode == 'replacement_val':
                return ReplacementValReplacementStrategy().replace(data, 'NaN', replacement_vals=replacement_value)

        except (TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def remove_outliers_from_quantitative_data(self, data, threshold, replacement_mode):
        raise NotImplementedError

    @overrides
    def remove_unwanted_labels(self, data, unwanted_labels, replacement_mode):
        try:
            if data is None or replacement_mode is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if replacement_mode == 'del_row':
                return DelRowReplacementStrategy().replace(data, 'unwanted_labels', unwanted_labels)


        except (TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def project_accelerometer_to_global_coordinates(self, data, target_columns, mode, support_columns=None):
        try:
            if data is None or target_columns is None or mode is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame):
                print(type(data))
                raise TypeError(type(data))

            if mode == 'mean_estimate_gravity':
                raise NotImplementedError

            if mode == 'known_gyroscope':
                raise NotImplementedError

            if mode == 'known_gravity':
                if len(target_columns) != len(support_columns):
                    raise TypeError(self.messages.PROVIDED_ARRAYS_DONT_MATCH_LENGTH.value)

                for ind, column in enumerate(target_columns):
                    data[column] = data[column] - data[support_columns[ind]]

                return data

            if mode == 'known_orientation':
                raise NotImplementedError


        except (TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def label_data(self, labels, data):
        try:
            if data is None or labels is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not (isinstance(data, pandas.DataFrame) and isinstance(labels, pandas.DataFrame)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if (len(labels) != len(data)):
                raise TypeError(self.messages.PROVIDED_FRAME_DOESNT_MATCH_DATA.value)

            return pandas.concat((labels, data), axis=1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def znormalize_quantitative_data(self, data, columns = None):
        try:
            if data is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if not all(column in data.keys() for column in columns):
                raise TypeError(self.messages.PROVIDED_ARRAY_DOESNT_MATCH_DATA.value)

            if columns is not None:
                data[columns] = (data[columns] - data[columns].mean()) / data[columns].std()
            else:
                data = (data - data.mean()) / data.std()
            return data

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def min_max_normalize_quantitative_data(self, data, columns=None):
        try:
            if data is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if not all(column in data.keys() for column in columns):
                raise TypeError(self.messages.PROVIDED_ARRAY_DOESNT_MATCH_DATA.value)

            if columns is not None:
                data[columns]=(data[columns]-data[columns].min())/(data[columns].max()-data[columns].min()) # to center around 0.0 substract 0.5
            else:
                data = (data - data.min()) / (data.max() - data.min()) # to center around 0.0 substract 0.5
            return data

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def re_represent_data(self, current_representation, target_representation, data):
        raise NotImplementedError

    @overrides
    def norm_quantitative_data(self, norm, represenation, data, columns = None):
        raise NotImplementedError



