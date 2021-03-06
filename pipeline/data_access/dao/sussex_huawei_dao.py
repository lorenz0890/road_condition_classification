from pipeline.data_access.dao.abstract_dao import DAO
from overrides import overrides
import os.path
from os import path
import traceback
import pandas
import numpy
import random


class SussexHuaweiDAO(DAO):

    def __init__(self):
        super().__init__()

    @overrides
    def read_data(self, file_path, column_names = None, use_rows = None, use_columns = None):
        """
        Load a dataset from disk. start_row and end_row can be used in case the data has to be loaded chunk-wise.
        :param file_path: string, path to data file
        :param column_names: list
        :param use_rows: (int, int), (start index, end_index)
        :param use_columns: list, column names
        :return: pandas.DataFrame, named clumns and loaded data
        """
        try:
            # 1. validate input
            if file_path is None: raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(file_path, str) : raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if not path.exists(file_path): raise FileNotFoundError(self.messages.FILE_NOT_FOUND.value)

            # 2. load data and return pandas, use only rows specified
            data =  None
            if use_rows is not None:
                data = pandas.read_csv(file_path, sep=" ", header=None,
                                       skiprows=(lambda i : i < use_rows[0] or i >= use_rows[0]+use_rows[1]),
                                       float_precision='high', dtype={0:numpy.int64})
            else:
                data = pandas.read_csv(file_path, sep=" ", header=None,
                                       float_precision='high', dtype={0:numpy.int64})

            # 3. remove unwated columns
            if use_columns is not None:
                if all(column in data.keys() for column in use_columns):
                    data = data[use_columns]
                else:
                    raise ValueError(self.messages.PROVIDED_ARRAY_DOESNT_MATCH_DATA.value)

            # 4. add label information
            if column_names is not None:
                if len(column_names) == len(data.columns):
                    data.columns = column_names
                else:
                    raise ValueError(self.messages.PROVIDED_ARRAY_DOESNT_MATCH_DATA.value)

            return data

        except (FileNotFoundError, ValueError, TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def bulk_read_data(self, file_path, identifiers, column_names, use_columns, check_distribution):
        """
        Load a multiple datasets from data directory on disk, return concatenated pandas dataframe.
        :param file_path: list(), paths to data files
        :param column_names: list(string)
        :param identifiers: list()
        :param use_columns: list(), column names
        :return: pandas.DataFrame, labeled and loaded data
        """
        try:
            # 1. validate input
            if file_path is None or identifiers is None or column_names is None or use_columns is None or check_distribution is None:
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(file_path, list): raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            all_labels = []
            all_data = []
            id = 0

            for identifier in identifiers:
                data_string = file_path[0].format(identifier)
                label_string = file_path[1].format(identifier)

                data = self.read_data(
                    data_string,
                    column_names=column_names[0],
                    use_columns=use_columns[0])  # 4,5,6,7,8,9,17,18,19

                data['id'] = id #Add id column
                all_data.append(data)

                labels = self.read_data(
                    label_string,
                    column_names=column_names[1],
                    use_columns=use_columns[1])

                labels['id'] = id
                all_labels.append(labels)

                id+=1

            if check_distribution:
                #2. shuffle trips of data until target label distribution is reached
                #or max number of trys.
                #TODO make configureable, use values from config and migrate to method
                distribution_ok = False
                max_trys = 20000
                trys_left = 20000

                #Get gobal distribution
                labels_dist = pandas.concat(all_labels, axis=0)
                labels_dist = labels_dist.loc[labels_dist['road_label'].isin([1,3])]
                labels_dist = labels_dist.loc[labels_dist['coarse_label'].isin([5])]
                upper, lower = 1.0, 0.0
                epsilon = 1.0/float(20000)
                delta = 1.0/float(20000)
                country, city = 0.0, 0.0
                if (3 in labels_dist['road_label'].value_counts().index and
                    1 in labels_dist['road_label'].value_counts().index
                ):
                    country = labels_dist['road_label'].value_counts()[3] / labels_dist.shape[0]
                    city = labels_dist['road_label'].value_counts()[1] / labels_dist.shape[0]
                    if city > country:
                        upper, lower = city, country
                    else:
                        upper, lower = country, city
                else:
                    #TODO raise error
                    pass

                print('True class distribution in all car trips, city:', city, 'country', country)
                print('Attempting to shuffle trips for representative train, '
                      'test, validation distribution with delta', epsilon)
                while not distribution_ok and trys_left > 0:
                    if trys_left%200 == 0:
                        print('Completion', round((1.0-trys_left/max_trys)*100,2), '%')

                    train_ok, test_ok, valid_ok = False, False, False
                    all_data_labels = list(zip(all_data, all_labels))
                    random.shuffle(all_data_labels)
                    all_data_shuffled, all_labels_shuffled = zip(*all_data_labels)

                    train = all_labels_shuffled[0:int(0.5*len(all_labels_shuffled))]
                    test =  all_labels_shuffled[int(0.5 * len(all_labels_shuffled)):int(0.75 * len(all_labels_shuffled))]
                    valid = all_labels_shuffled[int(0.75 * len(all_labels_shuffled)):]

                    train = train[0].loc[train[0]['road_label'].isin([1,3])]
                    train = train.loc[train['coarse_label'].isin([5])]
                    if 3 in train['road_label'].value_counts().index:
                        if lower-epsilon < train['road_label'].value_counts()[3]/train.shape[0] < upper+epsilon:
                            train_ok = True

                    test = test[0].loc[test[0]['road_label'].isin([1, 3])]
                    test = test.loc[test['coarse_label'].isin([5])]
                    if 3 in test['road_label'].value_counts().index:
                        if lower-epsilon < test['road_label'].value_counts()[3] / test.shape[0] < upper+epsilon:
                            test_ok = True

                    valid = valid[0].loc[valid[0]['road_label'].isin([1, 3])]
                    valid = valid.loc[valid['coarse_label'].isin([5])]
                    if 3 in valid['road_label'].value_counts().index:
                        if lower-epsilon < valid['road_label'].value_counts()[3] / valid.shape[0] < upper+epsilon:
                            valid_ok = True

                    if train_ok and test_ok and valid_ok:
                        distribution_ok = True
                        print('Country trips distributions in shuffled set with epsilon', epsilon)
                        print(train['road_label'].value_counts()[3] / train.shape[0])
                        print(test['road_label'].value_counts()[3] / test.shape[0])
                        print(valid['road_label'].value_counts()[3] / valid.shape[0])

                    trys_left -=1
                    epsilon += delta


            if len(all_labels) > 1 or len(all_data) > 1:
                #TODO: raise error if len not equal
                labels = pandas.concat(all_labels, axis=0)
                data = pandas.concat(all_data, axis=0)
                return labels, data
            else:
                return all_labels[0], all_data[0]

        except (ValueError, TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)


    @overrides
    def write_features(self, file_path, features):
        """
        Write extracted features to disk
        :param file_path:
        :param features:
        :return:
        """
        try:
            # 1. validate input
            if file_path is None: raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not isinstance(file_path, str): raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)
            if (not isinstance(features, pandas.DataFrame) and
                    not isinstance(features, pandas.core.frame.DataFrame) \
                    and not isinstance(features, pandas.core.series.Series)):
                raise TypeError(self.messages.ILLEGAL_ARGUMENT_TYPE.value)

            features.to_pickle(file_path)

        except (ValueError, TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def load_features(self, file_path):
        """
        Load extracted features from disk
        :param file_path:
        :return:
        """
        try:
            # 1. validate input
            if file_path is None: raise TypeError(self.messages.ILLEGAL_ARGUMENT_NONE_TYPE.value)
            if not path.exists(file_path): raise FileNotFoundError(self.messages.FILE_NOT_FOUND.value)

            data = pandas.read_pickle(file_path)
            return data

        except (FileNotFoundError, ValueError, TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)




    @overrides
    def write_model(self, file_path, mode, model):
        """
        Write trained features to disk
        :param file_path:
        :param model:
        :return:
        """
        raise NotImplementedError(self.messages.NOT_IMPLEMENTED.value)

    @overrides
    def load_model(self, file_path, mode):
        """
        Load trained model from disk
        :param file_path:
        :return:
        """
        raise NotImplementedError(self.messages.NOT_IMPLEMENTED.value)