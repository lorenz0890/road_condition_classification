from pipeline.feature_engineering.preprocessing.abstract_preprocessor import Preprocessor
from pipeline.feature_engineering.preprocessing.replacement_strategies.mean_replacement_strategy import MeanReplacementStrategy
from pipeline.feature_engineering.preprocessing.replacement_strategies.del_row_replacement_strategy import DelRowReplacementStrategy
from pipeline.feature_engineering.preprocessing.replacement_strategies.replacement_val_replacement_strategy import ReplacementValReplacementStrategy
from overrides import overrides
import traceback
import os
import pandas
from sklearn.decomposition import PCA
import numpy


class SussexHuaweiPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__()

    @overrides
    def segment_data(self, data, mode, label_column=None, args=None):
        """
        Segements a time series based on a label column, semantic segementation of a fixed interval.
        :param data:
        :param mode:
        :param label_column:
        :param args:
        :return:
        """
        try:
            if data is None or mode is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if mode == 'semantic':
                raise NotImplementedError(self.messages.NOT_IMPLEMENTED.value)

            if mode == 'labels':
                # 1. Select all data with desired label value
                data_segments = []
                for target_label in args:
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

            if mode == 'fixed_interval':
                segment_length = args[0]
                aggregate = args[1]
                exact_length = args[2]
                segments_aggregated = []
                split = lambda df, chunk_size : numpy.array_split(df, len(df) // chunk_size + 1, axis=0)

                # 1. Ensure index is datetime index and standardize type
                data.index = pandas.DatetimeIndex(data.index.astype('datetime64[1s]'))

                #2. Segment data
                segments = split(data, segment_length)
                if not exact_length:
                    for segment in segments:
                        segment.index = pandas.DatetimeIndex(segment.index.astype('datetime64[1s]'))
                    return segments

                #3. Remove segments that are too long or too short after splitting
                min_length_subsegements = []
                for segment in segments:
                    if segment.shape[0] == segment_length:
                        min_length_subsegements.append(segment)

                if not aggregate:
                    for segment in min_length_subsegements:
                        segment.index = pandas.DatetimeIndex(segment.index.astype('datetime64[1s]'))
                    return min_length_subsegements

                #3. Resample and aggregate data
                segments_combined = None
                for segment in min_length_subsegements:
                    segment = segment.reset_index()
                    segment.index = pandas.DatetimeIndex(segment.index.astype('datetime64[1s]'))
                    segment = self.resample_quantitative_data(segment,
                                                              freq="{}s".format(segment_length),
                                                              mode = 'mean')

                    if segments_combined is None:
                        segments_combined = segment
                    else:
                        segments_combined = pandas.concat([segments_combined, segment], axis=0)

                if segments_combined is not None:
                    segments_combined = segments_combined.reset_index()
                    segments_combined.index = pandas.DatetimeIndex(
                        segments_combined.index.astype('datetime64[1s]'))
                    segments_aggregated.append(segments_combined)

                return segments_aggregated

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)


        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def de_segment_data(self, data_segments, selected_columns=None, axis = 0):
        """
        Desegements as time series.
        :param data_segments:
        :param selected_columns:
        :param axis:
        :return:
        """
        try:
            data = None
            for ind in range(len(data_segments)):
                if data is None:
                    data = data_segments[ind][selected_columns]
                else:
                    data = pandas.concat([data, data_segments[ind][selected_columns]], axis=axis)
                    data = data.reset_index(drop=True)

            return data

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)


        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)


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

            raise ValueError(self.messages.PROVIDED_MODE_DOESNT_EXIST.value)

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)


        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def remove_outliers_from_quantitative_data(self, data, replacement_mode, columns, quantile = None, threshold = None):
        """
        Removes outlieres either based on quantile or a threshold value.
        :param data:
        :param replacement_mode:
        :param columns:
        :param quantile:
        :param threshold:
        :return:
        """
        try:
            if data is None or replacement_mode is None or columns is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame) or not isinstance(columns, list) or not isinstance(replacement_mode, str):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if len(columns) < 1:
                raise ValueError(self.messages.PROVIDED_ARRAY_DOESNT_MATCH_DATA.value)


            if replacement_mode == 'quantile':
                # Source for next 7 lines of code after comment:
                # https://nextjournal.com/schmudde/how-to-remove-outliers-in-data
                for column in columns:
                    not_outliers = data[column].between(
                            data[column].quantile(1.0 - quantile),
                            data[column].quantile(quantile)
                        )

                    data[column] = data[column][not_outliers]
                    index_names = data[~not_outliers].index
                    data.drop(index_names, inplace=True)

                old_index = data.index
                data = data.reset_index(drop=False)
                data = data.set_index(old_index)

                return data

            if replacement_mode == 'threshold':
                raise NotImplementedError(self.messages.NOT_IMPLEMENTED.value)

            raise ValueError(self.messages.PROVIDED_MODE_DOESNT_EXIST.value)

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)


        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)


    @overrides
    def resample_quantitative_data(self, data, freq, mode = None):
        """
        Resamples quantitative data.
        :param data:
        :param freq:
        :param mode:
        :return:
        """
        # Source:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        # https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html
        try:
            if data is None or freq is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame) or not isinstance(freq, str):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if mode == 'mean' or mode is None:
                return data.resample(freq).mean()

            if mode == 'sum':
               return data.resample(freq).sum()

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def convert_unix_to_datetime(self, data, column, unit):
        """
        Converts unix time stamps to date time.
        :param data:
        :param column:
        :param unit:
        :return:
        """
        # Source:
        # https://stackoverflow.com/questions/19231871/convert-unix-time-to-readable-date-in-pandas-dataframe
        # https://stackoverflow.com/questions/42698421/pandas-to-datetime-from-milliseconds-produces-incorrect-datetime
        try:
            if data is None or column is None or unit is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame) or not isinstance(column, str) or not isinstance(unit, str):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            data[column] = pandas.to_datetime(data[column], unit=unit)
            return data

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def remove_unwanted_labels(self, data, unwanted_labels, replacement_mode):
        """
        Remove rows that have an unwanted label.
        :param data:
        :param unwanted_labels:
        :param replacement_mode:
        :return:
        """
        try:
            if data is None or replacement_mode is None or unwanted_labels is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame) or not isinstance(unwanted_labels, list) or not isinstance(replacement_mode, str):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if replacement_mode == 'del_row':
                return DelRowReplacementStrategy().replace(data, 'unwanted_labels', unwanted_labels)

            raise ValueError(self.messages.PROVIDED_MODE_DOESNT_EXIST.value)

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def project_accelerometer_to_global_coordinates(self, data, target_columns, mode, args=None):
        """
        Project accelerometer data from local vehicle coordinates to a global coordinate system.
        :param data:
        :param target_columns:
        :param mode:
        :param args:
        :return:
        """
        try:
            if data is None or target_columns is None or mode is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame) or not isinstance(mode, str) or not isinstance(target_columns, list):
                raise TypeError(type(data))

            if mode == 'mean_estimate_gravity':
                raise NotImplementedError(self.messages.NOT_IMPLEMENTED.value)

            if mode == 'gyroscope':
                raise NotImplementedError(self.messages.NOT_IMPLEMENTED.value)

            if mode == 'gravity':
                if len(target_columns) != len(args):
                    raise TypeError(self.messages.PROVIDED_ARRAYS_DONT_MATCH_LENGTH.value)

                for ind, column in enumerate(target_columns):
                    data[column] = data[column] - data[args[ind]]

                return data

            if mode == 'orientation':
                if len(target_columns)+1 != len(args):
                    raise TypeError(self.messages.PROVIDED_ARRAYS_DONT_MATCH_LENGTH.value)

                # Source for theory behind below calculation
                # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
                # https://en.wikipedia.org/wiki/Homogeneous_coordinates
                # #https://stackoverflow.com/questions/2422750/in-opengl-vertex-shaders-what-is-w-and-why-do-i-divide-by-it
                for ind, column in enumerate(target_columns):
                    data[column] = data[column] * (data[args[ind]] / data[args[3]])

                return data

            raise ValueError(self.messages.PROVIDED_MODE_DOESNT_EXIST.value)

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def label_data(self, labels, data):
        """
        Combines labels vector and data matrix.
        :param labels:
        :param data:
        :return:
        """
        try:
            if data is None or labels is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not (isinstance(data, pandas.DataFrame) and isinstance(labels, pandas.DataFrame)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if (len(labels) != len(data)):
                raise TypeError(self.messages.PROVIDED_FRAME_DOESNT_MATCH_DATA.value)

            return pandas.concat((labels, data), axis=1)

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def znormalize_quantitative_data(self, data, columns = None, mean = None, std = None):
        """
        Apply z-normalization to a data set.
        :param data:
        :param columns:
        :param mean:
        :param std:
        :return:
        """
        try:
            if data is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if not all(column in data.keys() for column in columns):
                raise TypeError(self.messages.PROVIDED_ARRAY_DOESNT_MATCH_DATA.value)

            if mean is None and std is None:
                if columns is not None:
                    mean = data[columns].mean()
                    std = data[columns].std()
                    data[columns] = (data[columns] - data[columns].mean()) / data[columns].std()
                else:
                    mean = data.mean()
                    std = data.std()
                    data = (data - data.mean()) / data.std()
            elif mean is not None and std is not None:
                if columns is not None:
                    data[columns] = (data[columns] - mean) / std
                else:
                    data = (data - mean) / std
            else:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            return data, mean, std

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)


        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def min_max_normalize_quantitative_data(self, data, columns=None):
        """
        Apply min-max-normalization to a data set.
        :param data:
        :param columns:
        :return:
        """
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

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def re_represent_data(self, current_representation, target_representation, data):
        """
        Change representation of a data set.
        :param current_representation:
        :param target_representation:
        :param data:
        :return:
        """
        raise NotImplementedError(self.messages.NOT_IMPLEMENTED.value)

    @overrides
    def reduce_quantitativ_data_dimensionality(self, data, mode, reduced_column_name = 'reduced', columns = None):
        """
        Apply a dimensionality reduction technique to a data set.
        :param data:
        :param mode:
        :param reduced_column_name:
        :param columns:
        :return:
        """
        try:
            if data is None or mode is None or reduced_column_name is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame) or not isinstance(mode, str) or not isinstance(reduced_column_name, str):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if mode == 'euclidean':
                # Source:
                # https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/
                # https://www.google.com/search?client=ubuntu&channel=fs&q=euclidean+norm&ie=utf-8&oe=utf-8
                # https://stackoverflow.com/questions/54260920/combine-merge-dataframes-with-different-indexes-and-different-column-names
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
                reduced = data[columns].apply(numpy.square, axis=1)[columns].sum(axis=1).apply(numpy.sqrt, axis=1)
                old_index = data.index
                data = pandas.concat([data, reduced], axis=1)
                data = data.rename(columns={0: reduced_column_name})
                data = data.reset_index(drop=True)
                data = data.set_index(old_index)
                return data

            if mode == 'manhatten':
                reduced = data[columns].sum(axis=1)[columns].apply(numpy.abs, axis=1)
                old_index = data.index
                data = pandas.concat([data, reduced], axis=1)
                data = data.rename(columns={0: reduced_column_name})
                data = data.reset_index(drop=True)
                data = data.set_index(old_index)
                return data

            if mode == 'pca':
                # Source:
                # https://stackoverflow.com/questions/23282130/principal-components-analysis-using-pandas-dataframe
                # https://stackoverflow.com/questions/54260920/combine-merge-dataframes-with-different-indexes-and-different-column-names
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
                # https://en.wikipedia.org/wiki/Principal_component_analysis
                pca = PCA(n_components=1)
                pca.fit(data[columns])
                reduced = pandas.DataFrame((numpy.dot(pca.components_, data[columns].T).T))
                reduced = reduced.rename(columns={0:reduced_column_name})
                reduced = reduced.reset_index(drop=True)
                old_index = data.index
                data = data.reset_index(drop=True)
                data = pandas.concat([data, reduced], axis=1)
                data = data.set_index(old_index)
                return data

            raise ValueError(self.messages.PROVIDED_MODE_DOESNT_EXIST.value)

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def encode_categorical_features(self, data, mode, columns, encoding_function):
        """
        Encode categorical features using an encoding function.
        :param data:
        :param mode:
        :param columns:
        :param encoding_function:
        :return:
        """
        try:
            if data is None or mode is None or columns is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(data, pandas.DataFrame) or not isinstance(mode, str) or not isinstance(
                    columns, list):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            if mode == 'custom_function':
                if encoding_function is None:
                    raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)

                for column in columns:
                    data[column] = encoding_function(data[column])
                return data


            raise ValueError(self.messages.PROVIDED_MODE_DOESNT_EXIST.value)

        except (TypeError, NotImplementedError, ValueError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def inference_split_process(self, data, config, meta_data):
        """
        Apply all preprocessing steps necessary for inference.
        :param data: pandas.DataFrame
        :param params: List
        :return: pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
        """
        print('Fetch params')
        acelerometer_columns = [config['data_set_column_names'][1:][0], config['data_set_column_names'][1:][1], config['data_set_column_names'][1:][2]]
        freq = config['pre_proc_resample_freq']  # '1000ms'
        mean_train = meta_data['mean_train']
        std_train = meta_data['std_train']


        print('Convert time unit, remove nans')
        data = self.convert_unix_to_datetime(data, column='time', unit='ms')
        data = self.remove_nans(data, replacement_mode='del_row')

        data.set_index(data['time'], drop=True, inplace=True)
        print('Resample')
        data = self.resample_quantitative_data(data,
                                        freq=freq)  # 8000 1.25 Hz

        print('Dimensionality reduction')
        data = self.reduce_quantitativ_data_dimensionality(
            data=data,
            mode=config['feature_eng_dim_reduction_type'],  # works better than euclidean for motif
            columns=acelerometer_columns,
            reduced_column_name='acceleration_abs'
        )

        print('Normalizing, outlier removal')
        selected_columns = ['acceleration_abs']
        data, mean, std = self.znormalize_quantitative_data(data, selected_columns, mean_train, std_train)
        data = self.remove_outliers_from_quantitative_data(
            data,
            replacement_mode='quantile',
            columns=selected_columns,
            quantile=0.99  # current run @0.95 for classical approach via TS Fresh
        )[:-1]

        return data


    @overrides
    def training_split_process(self, data, config, labels):
        """
        Apply all preprocessing steps necessary for training.
        :param data: pandas.DataFrame
        :param params: List
        :return: pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
        """
        print('Fetch params')
        #print(params)
        labels = labels
        test_sz = config['pre_proc_test_sz']
        train_sz = config['pre_proc_training_sz']
        valid_sz = config['pre_proc_validation_sz']
        #acelerometer_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']
        acelerometer_columns = [config['data_set_column_names'][1:][0], config['data_set_column_names'][1:][1], config['data_set_column_names'][1:][2]]
        selected_coarse_labels = config['pre_proc_movement_type_label'] #[5]
        selected_road_labels = config['pre_proc_road_type_label'] #[1, 3]
        freq = config['pre_proc_resample_freq'] #'1000ms'

        print('Convert time unit, label data, remove nans')
        data = self.convert_unix_to_datetime(data, column = 'time', unit = 'ms')
        data = self.label_data(data, labels)
        data = self.remove_nans(data, replacement_mode='del_row')

        print('Train, Test, Validation split')
        data_len = data.shape[0]
        test_len = int(data_len * test_sz)
        train_len = int(data_len * train_sz)
        valid_len = int(data_len * valid_sz)
        data_train, data_test_valid = data.head(train_len), data.tail(test_len+valid_len)
        data_test = data_test_valid.head(test_len)
        data_valid = data_test_valid.tail(valid_len)

        print('Segment by labels')
        #Segment Train
        car_train_segments = self.segment_data(data_train, mode='labels',
                                                 label_column='coarse_label',
                                                 args=selected_coarse_labels)
        data_train_segments = []
        for car_segment in car_train_segments:
            road_segments = self.segment_data(car_segment, mode='labels',
                                                      label_column='road_label',
                                                      args=selected_road_labels
                                                      )
            for road_segment in road_segments:
                data_train_segments.append(road_segment)


        #Segment Test
        car_test_segments = self.segment_data(data_test, mode='labels',
                                               label_column='coarse_label',
                                               args=selected_coarse_labels)
        data_test_segments = []
        for car_segment in car_test_segments:
            road_segments = self.segment_data(car_segment, mode='labels',
                                              label_column='road_label',
                                              args=selected_road_labels
                                              )
            for road_segment in road_segments:
                data_test_segments.append(road_segment)

        #Segment Valid
        car_valid_segments = self.segment_data(data_valid, mode='labels',
                                              label_column='coarse_label',
                                              args=selected_coarse_labels)
        data_valid_segments = []
        for car_segment in car_valid_segments:
            road_segments = self.segment_data(car_segment, mode='labels',
                                              label_column='road_label',
                                              args=selected_road_labels
                                              )
            for road_segment in road_segments:
                data_valid_segments.append(road_segment)

        print('Resample')
        #Train
        for ind in range(len(data_train_segments)):
            data_train_segments[ind] = data_train_segments[ind].set_index('time')
            data_train_segments[ind] = self.resample_quantitative_data(data_train_segments[ind],
                                                                         freq=freq)  # 8000 1.25 Hz

        #Test
        for ind in range(len(data_test_segments)):
            data_test_segments[ind] = data_test_segments[ind].set_index('time')
            data_test_segments[ind] = self.resample_quantitative_data(data_test_segments[ind],
                                                                         freq=freq)

        #Valid
        for ind in range(len(data_valid_segments)):
            data_valid_segments[ind] = data_valid_segments[ind].set_index('time')
            data_valid_segments[ind] = self.resample_quantitative_data(data_valid_segments[ind],
                                                                      freq=freq)

        print('Dimensionality reduction')
        #Train
        for ind in range(len(data_train_segments)):
            data_train_segments[ind] = self.reduce_quantitativ_data_dimensionality(
                data=data_train_segments[ind],
                mode=config['feature_eng_dim_reduction_type'],  # works better than euclidean for motif
                columns=acelerometer_columns,
                reduced_column_name='acceleration_abs'
            )

        #Test
        for ind in range(len(data_test_segments)):
            data_test_segments[ind] = self.reduce_quantitativ_data_dimensionality(
                data=data_test_segments[ind],
                mode=config['feature_eng_dim_reduction_type'],  # works better than euclidean for motif
                columns=acelerometer_columns,
                reduced_column_name='acceleration_abs'
            )

        #Valid
        for ind in range(len(data_valid_segments)):
            data_valid_segments[ind] = self.reduce_quantitativ_data_dimensionality(
                data=data_valid_segments[ind],
                mode=config['feature_eng_dim_reduction_type'],  # works better than euclidean for motif
                columns=acelerometer_columns,
                reduced_column_name='acceleration_abs'
            )

        print('Normalizing, outlier removal')
        #Train
        selected_columns = ['acceleration_abs',
                            'road_label', 'id']  # 'acceleration_abs'
        data_train = self.de_segment_data(data_train_segments, selected_columns)
        data_train, mean_train, std_train = self.znormalize_quantitative_data(data_train, selected_columns[:-2])

        data_train = self.remove_outliers_from_quantitative_data(
            data_train,
            replacement_mode='quantile',
            columns=selected_columns[:-2],
            quantile=0.99  # current run @0.95 for classical approach via TS Fresh
        )[:-2]

        #Test
        data_test = self.de_segment_data(data_test_segments, selected_columns)
        data_test, mean_test, std_test = self.znormalize_quantitative_data(data_test,
                                                                          selected_columns[:-2],
                                                                          mean_train, std_train)

        data_test = self.remove_outliers_from_quantitative_data(
            data_test,
            replacement_mode='quantile',
            columns=selected_columns[:-2],
            quantile=0.99  # current run @0.95 for classical approach via TS Fresh
        )[:-2]

        #Valid
        data_valid = self.de_segment_data(data_valid_segments, selected_columns)
        data_valid, mean_valid, std_valid = self.znormalize_quantitative_data(data_valid,
                                                                           selected_columns[:-2],
                                                                           mean_train, std_train)

        data_valid = self.remove_outliers_from_quantitative_data(
            data_valid,
            replacement_mode='quantile',
            columns=selected_columns[:-2],
            quantile=0.99  # current run @0.95 for classical approach via TS Fresh
        )[:-2]

        data_train = data_train.loc[:, ~data_train.columns.duplicated()]
        data_test = data_test.loc[:, ~data_test.columns.duplicated()]
        data_valid = data_valid.loc[:, ~data_valid.columns.duplicated()]

        #print('Rolling mean smoothing')
        #data_train['acceleration_abs'] = data_train['acceleration_abs'].rolling(5, min_periods=1, win_type='gaussian').sum(std=3) #TODO make configureable
        #data_test['acceleration_abs'] = data_test['acceleration_abs'].rolling(5, min_periods=1, win_type='gaussian').sum(std=3)
        #data_valid['acceleration_abs'] = data_valid['acceleration_abs'].rolling(5, min_periods=1, win_type='gaussian').sum(std=3)
        #data_train = self.remove_nans(data_train, replacement_mode='del_row')
        #data_test = self.remove_nans(data_test, replacement_mode='del_row')
        #data_valid = self.remove_nans(data_valid, replacement_mode='del_row')
        #print(data_train.head(100))
        return data_train, mean_train, std_train, data_test, data_valid