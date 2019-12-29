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
        mp = matrixProfile.scrimp_plus_plus(data[args[3]].values, args[0]) #6 or 32
        mtfs, motif_d = motifs.motifs(data[args[3]].values, mp, max_motifs=args[1], radius=args[2])  # 2, 23

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
                    attr_vec[count + pos][0] = x
                    if args[1] == 2:
                        attr_vec[count + pos][1] = i

                count += args[0]
            i += 1.0

        X = attr_vec#.transpose()
        X = pandas.DataFrame(X)

        return X#, y