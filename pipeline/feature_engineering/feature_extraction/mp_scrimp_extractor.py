from pipeline.feature_engineering.feature_extraction.abstract_extractor import Extractor
from overrides import overrides
from matrixprofile import *
#from multiprocessing import Pool
import multiprocessing as mp
import numpy
import pandas

class MPScrimpExtractor(Extractor):

    def __init__(self):
        super().__init__()

    @overrides
    def extract_features(self, data, args = None):
        """
        Extract features
        Source:
        https://github.com/target/matrixprofile-ts/blob/master/docs/examples/Motif%20Discovery.ipynb
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

    @overrides
    def extract_select_features(self, data, args=None):
        """
        Find different combinations of motifs dependent on hyperparameters
        Sources for mp:
        https://stackoverflow.com/questions/29009790/python-how-to-do-multiprocessing-inside-of-a-class
        https://stackoverflow.com/questions/8953119/python-waiting-for-external-launched-process-finish
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """

        num_processors = 32  # create a pool of processors
        #p = Pool(processes=num_processors)  # get them to work in parallel#
        #output = p.map(worker, [i for i in range(0, 29)])  # 6*5 = 30

        output = {}
        processes = []
        for i in range(num_processors):
            p = mp.Process(target=self.__worker, args=(i, data, output))
            processes.append(p)

        [x.start() for x in processes]
        [x.join() for x in processes]
        
        result_list = []
        result_list.append(output[0].keys())
        for elem in output:
            templist = []
            for key in elem.keys():
                templist.append(elem[key])
            result_list.append(templist)

        return result_list

    def __worker(self, i, data, output):
        combis = []
        radii = [8, 12, 16, 20, 24, 32]  # 6
        lengths = [6, 12, 18, 24, 32]  # 5
        for radius in radii:
            for length in lengths:
                combi = [radius, length]
                combis.append(combi)
        print("Motif extraction worker no: {0} length: {1}, radius: {2}".format(i, combis[i][1], combis[i][0]))
        X_indices = self.extract_features(data=data,
                                          args=[combis[i][1], 2, combis[i][0], 'acceleration_abs'])
        X = self.select_features(data=data,
                                 args=[combis[i][1], 2, X_indices, 'acceleration_abs'])
        y = self.select_features(data=data,
                                 args=[combis[i][1], 1, X_indices, 'road_label'])

        print("Motif extraction worker no: {0} returned".format(i))
        output[i] = {
            'X': X,
            'y': y,
            'radius': combis[i][0],
            'length': combis[i][1],
            'motifs': 2
        }