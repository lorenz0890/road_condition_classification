import unittest
import numpy
from feature_engineering.dao.sussex_huawei_dao import SussexHuaweiDAO
from feature_engineering.preprocessing.sussex_huawei_preprocessor import SussexHuaweiPreprocessor

class TestSussexHuaweiPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dao = SussexHuaweiDAO()
        cls.preprocessor = SussexHuaweiPreprocessor()
        label_column_names = ['coarse_label', 'fine_label', 'road_label']
        cls.labels = cls.dao.read_data(
            '/home/lorenz/PycharmProjects/rctc_pipeline/data_sets/sussex_huawei/User1/220617/Label.txt', #TODO: Pack in config/.env
            column_names=label_column_names, use_columns=[1, 2, 3])

        cls.data_column_names = ['time', 'acceleration_x', 'acceleration_y', 'acceleration_z', #TODO: Pack in config/.env
                             'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
                             'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                             'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
                             'gravity_x', 'gravity_y', 'gravity_z',
                             'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
                             ]
        cls.data = cls.dao.read_data(
            '/home/lorenz/PycharmProjects/rctc_pipeline/data_sets/sussex_huawei/User1/220617/Hand_Motion.txt', #TODO: Pack in config/.env
            column_names=cls.data_column_names, use_columns=[0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])


    def test_remove_nans_mean(self):
        data = self.preprocessor.remove_nans(self.data, replacement_mode='mean')
        self.assertTrue(not(True in data.isnull()))

    def test_remove_nans_del_row(self):
        data = self.preprocessor.remove_nans(self.data, replacement_mode='del_row')
        self.assertTrue(not(True in data.isnull()))

    def test_remove_nans_replace_val(self):
        data = self.preprocessor.remove_nans(self.data, replacement_mode='replacement_val', replacement_value=1)
        self.assertTrue(not(True in data.isnull()))

    def test_remove_unwanted_del_row(self):
        unwanted_labels = {
            'coarse_label' : [2, 3, 4, 6, 7, 8, 0],
            'road_label' : [2, 4, 0]
        }
        data = self.preprocessor.label_data(self.data, self.labels)
        data = self.preprocessor.remove_nans(data, replacement_mode='del_row')
        num_rows_before_removal = data.shape[0]
        data = self.preprocessor.remove_unwanted_labels(data, replacement_mode='del_row', unwanted_labels=unwanted_labels)
        self.assertTrue(0 < data.shape[0] < num_rows_before_removal)

    def test_label_data(self):
        data = self.preprocessor.label_data(self.data, self.labels)
        self.assertTrue(all(column in data for column in self.labels.keys()))

    def test_znormalize_quantitative_data(self):
        unwanted_labels = {
            'coarse_label': [2, 3, 4, 6, 7, 8, 0],
            'road_label': [2, 4, 0]
        }
        data = self.preprocessor.label_data(self.data, self.labels)
        data = self.preprocessor.remove_nans(data, replacement_mode='del_row')
        data = self.preprocessor.remove_unwanted_labels(data, replacement_mode='del_row',
                                                        unwanted_labels=unwanted_labels)

        columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']
        data = self.preprocessor.znormalize_quantitative_data(data, columns)
        self.assertTrue(all(data[columns].mean() < 1e-10))

    def test_min_max_normalize_quantitative_data(self):
        unwanted_labels = {
            'coarse_label': [2, 3, 4, 6, 7, 8, 0],
            'road_label': [2, 4, 0]
        }
        data = self.preprocessor.label_data(self.data, self.labels)
        data = self.preprocessor.remove_nans(data, replacement_mode='del_row')
        data = self.preprocessor.remove_unwanted_labels(data, replacement_mode='del_row',
                                                        unwanted_labels=unwanted_labels)

        columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']
        data = self.preprocessor.min_max_normalize_quantitative_data(data, columns)
        self.assertTrue(all(data[columns].max() <= 1.0) and all(0.0 <= data[columns].min()))

    def test_convert_unix_to_datetime(self):
        data = self.preprocessor.convert_unix_to_datetime(self.data, column = 'time', unit = 'ms')
        self.assertTrue(not isinstance(data['time'].iloc[0], numpy.int64))

    def test_normalization(self):
        accelerometer_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']
        gravity_columns = ['gravity_x', 'gravity_y', 'gravity_z']
        orientation_columns = ['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']

        #Not ideal to combine multiply tests into one test case. TODO: Split into single test cases
        data = self.preprocessor.project_accelerometer_to_global_coordinates(
            self.data,
            mode='gravity',
            target_columns=accelerometer_columns,
            support_columns=gravity_columns)

        data = self.preprocessor.project_accelerometer_to_global_coordinates(
            data,
            mode='orientation',
            target_columns=accelerometer_columns,
            support_columns=orientation_columns)

        data = self.preprocessor.znormalize_quantitative_data(data, self.data_column_names[1:])
        self.assertTrue(all(numpy.abs(data[accelerometer_columns].mean()) < 10e-5))

    def test_segment_data(self):
        selected_coarse_labels = [5]
        data = self.preprocessor.label_data(self.data, self.labels)
        data = self.preprocessor.segment_data(data, mode='labels',
                                  label_column='coarse_label',
                                  support=selected_coarse_labels)

        self.assertTrue(len(data) > 1 and segment['coarse_label'] in selected_coarse_labels for segment in data)

    def test_resample_quantitative_data(self):
        self.data = self.preprocessor.label_data(self.data, self.labels)
        self.data = self.preprocessor.remove_nans(self.data, replacement_mode='del_row')
        data = self.preprocessor.convert_unix_to_datetime(self.data, column='time', unit='ms')
        data = data.set_index('time')
        data = self.preprocessor.resample_quantitative_data(data, freq='100ms')
        self.assertTrue(0 < len(data) < len(self.data))

if __name__ == '__main__':
    unittest.main()