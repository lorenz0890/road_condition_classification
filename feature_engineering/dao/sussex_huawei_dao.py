from feature_engineering.dao.abstract_dao import DAO
from overrides import overrides
import os.path
from os import path
import traceback
import pandas

class SussexHuaweiDAO(DAO):

    def __init__(self):
        super().__init__()

    @overrides
    def read_data(self, file_path, column_names = None, use_rows = None, use_columns = None):
        """
        Load a dataset from disk. start_row and end_row can be used in case the data has to be loaded chunk-wise.
        :param file_path: string, path to data file
        :param column_names: array[string]
        :param use_rows: (int, int), (start index, end_index)
        :param use_columns: array[], column names
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
                data = pandas.read_csv(file_path, sep=" ", header=None, skiprows=(lambda i : i < use_rows[0] or i >= use_rows[0]+use_rows[1]))
            else:
                data = pandas.read_csv(file_path, sep=" ", header=None,)

            # 3. remove unwated columns
            if use_columns is not None:
                if all(column in data.keys() for column in use_columns):
                    data = data[use_columns]
                else:
                    raise IndexError(self.messages.PROVIDED_ARRAY_DOESNT_MATCH_DATA.value)

            # 4. add label information
            if column_names is not None:
                if len(column_names) == len(data.columns):
                    data.columns = column_names
                else:
                    raise IndexError(self.messages.PROVIDED_ARRAY_DOESNT_MATCH_DATA.value)

            return data

        except (FileNotFoundError, IndexError, TypeError):
            self.logger.error(traceback.format_exc())
            os._exit(1)

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)


    @overrides
    def write_features(self, file_path, data_dict):
        """
        Write extracted features to disk
        :param file_path:
        :param data_dict:
        :return:
        """
        raise NotImplementedError

    @overrides
    def load_features(self, file_path):
        """
        Load extracted features from disk
        :param file_path:
        :return:
        """
        raise NotImplementedError