import unittest
from pipeline.data_access.dao.sussex_huawei_dao import SussexHuaweiDAO

class TestSussexHuaweiDao(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dao = SussexHuaweiDAO()

    def test_raw_read_data_success(self):
        data = self.dao.read_data('/home/lorenz/PycharmProjects/rctc_pipeline/data_sets/sussex_huawei/User1/220617/Hand_Motion.txt')
        self.assert_(len(data) > 0)

    def test_raw_read_data_clip_rows(self):
        data = self.dao.read_data(
            '/home/lorenz/PycharmProjects/rctc_pipeline/data_sets/sussex_huawei/User1/220617/Hand_Motion.txt',
            use_rows=[0,10])
        self.assertTrue(len(data) == 10)

    def test_raw_read_data_clip_columns(self):
        data = self.dao.read_data(
            '/home/lorenz/PycharmProjects/rctc_pipeline/data_sets/sussex_huawei/User1/220617/Hand_Motion.txt',
            use_columns=[1,2,3])
        self.assertTrue(len(data.keys()) == 3)

    def test_raw_read_data_column_names(self):
        column_names = ['a', 'b', 'c']
        data = self.dao.read_data(
            '/home/lorenz/PycharmProjects/rctc_pipeline/data_sets/sussex_huawei/User1/220617/Hand_Motion.txt',
            column_names = column_names, use_columns=[1, 2, 3], use_rows=[900, 10])
        self.assertTrue(all(column_name in data.columns for column_name in column_names))


if __name__ == '__main__':
    unittest.main()