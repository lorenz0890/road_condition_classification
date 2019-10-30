import unittest
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

        data_column_names = ['acceleration_x', 'acceleration_y', 'acceleration_z', #TODO: Pack in config/.env
                             'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
                             'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                             'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
                             'gravity_x', 'gravity_y', 'gravity_z',
                             'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
                             ]
        cls.data = cls.dao.read_data(
            '/home/lorenz/PycharmProjects/rctc_pipeline/data_sets/sussex_huawei/User1/220617/Hand_Motion.txt', #TODO: Pack in config/.env
            column_names=data_column_names, use_columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])


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



if __name__ == '__main__':
    unittest.main()