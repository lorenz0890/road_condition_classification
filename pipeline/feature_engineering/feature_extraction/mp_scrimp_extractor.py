from pipeline.feature_engineering.feature_extraction.abstract_extractor import Extractor
from overrides import overrides
from matrixprofile import *
import numpy
import pandas

class MPScrimpExtractor(Extractor):

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
        mp = matrixProfile.scrimp_plus_plus(data[args[3]].values[:1500], args[0]) #6 or 32
        mtfs, motif_d = motifs.motifs(data[args[3]].values[:1500], mp, max_motifs=args[1], radius=args[2])  # 2, 23

        return mtfs

    @overrides
    def select_features(self, data, args=None):
        """
        Select features
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        mtfs = args[2]
        sz = len([item for sublist in mtfs for item in sublist]) * args[0]
        attr_vec = numpy.ndarray(shape=(sz, args[1]), dtype=float) #3
        # print(mtfs)
        count = 0
        i = 1.0
        for motif in mtfs:
            for index in motif:  # ['acceleration_abs', 'road_label']
                elem = numpy.array(data[args[3]].values[index:index + args[0]])
                for pos, x in enumerate(elem):
                    attr_vec[count + pos][0] = x[0]
                    attr_vec[count + pos][1] = i
                    #attr_vec[count + pos][2] = x[1]

                count += args[0]
            i += 1.0

        X = attr_vec.transpose()[:2].transpose()
        y = attr_vec.transpose()[2]
        # print(X_train.shape)
        X = pandas.DataFrame(X)
        y = pandas.DataFrame(y)

        return X, y