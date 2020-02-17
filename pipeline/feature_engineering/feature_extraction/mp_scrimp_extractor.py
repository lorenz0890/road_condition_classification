from pipeline.feature_engineering.feature_extraction.abstract_extractor import Extractor
from overrides import overrides
from matrixprofile import *
from IPython.display import clear_output
#from multiprocessing import Pool
#import pathos.multiprocessing
import gc
import traceback
import os
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
        try:
            mp = matrixProfile.scrimp_plus_plus(data[args[3]].values, args[0]) #6 or 32
            mtfs, motif_d = motifs.motifs(data[args[3]].values, mp, max_motifs=args[1], radius=args[2])  # 2, 23

            return mtfs, motif_d

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)


    @overrides
    def select_features(self, data, args=None):
        """
        Select features by first finding motifs in time series and tagging them.
        Since scrimp returns motif sorted by distances, the top motifs should
        receive the same tags in training and inference sets if they are from the same source.
        Finally PAA is used for whole motifs found in data via std, mean, min and max.
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        try:
            mtfs = args[2]
            motif_d = args[4]
            sz = len([item for sublist in mtfs for item in sublist]) * args[0]
            attr_vec = numpy.ndarray(shape=(sz, args[1]), dtype=float) #3
            count = 0
            tag = 1.0
            for i, motif in enumerate(mtfs):
                for index in motif:  # ['acceleration_abs', 'road_label']
                    elem = numpy.array(data[args[3]].values[index:index + args[0]])
                    for pos, x in enumerate(elem):
                        attr_vec[count + pos][0] = x
                        if args[1] >= 2:
                            attr_vec[count + pos][1] = tag #Add tag of found motif
                        if args[1] >= 3:
                            attr_vec[count + pos][2] = elem.std() #Add std of found motif
                        if args[1] >= 4:
                            attr_vec[count + pos][3] = numpy.amin(elem) #Add min of found motif
                        if args[1] >= 5:
                            attr_vec[count + pos][4] = numpy.amax(elem) #Add max of found motif
                        if args[1] >= 6:
                            attr_vec[count + pos][5] = motif_d[i] #Add 0.25 percentile

                    count += args[0]
                tag += 1.0

            X = attr_vec#.transpose()
            X = pandas.DataFrame(X)
            if args[1] == 1: #If args[1] = 1 then we are selecting relevant labels
                X = X.groupby(X.index // args[0]).first()
            if args[1] >= 2: #If args[1] = 2 then we are selecting relevant features
                X = X.groupby(X.index // args[0]).mean()

            X = X.dropna()
            return X#, y

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    @overrides
    def extract_select_training_features(self, data, args=None):
        """
        Find different combinations of motifs dependent on hyperparameters
        Sources for mp:
        https://stackoverflow.com/questions/29009790/python-how-to-do-multiprocessing-inside-of-a-class
        https://stackoverflow.com/questions/8953119/python-waiting-for-external-launched-process-finish
        https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """
        try:

            manager = mp.Manager()
            output = manager.dict()

            radii = args[0] #[8, 12, 16, 20, 24, 32]  # 6 TODO: if args is None use these values
            lengths = args[1] #[6, 12, 18, 24, 32]  # 5
            num_tasks = len(radii)*len(lengths)
            task_id = 0
            num_processors = 32# args[2]
            while task_id < num_tasks:
                processes = []
                for i in range(num_processors):
                    if num_processors >= 0 and i < num_tasks:#TODO: Consider removal of first condition
                        p = mp.Process(target=self.__extract_select_training_worker, args=(task_id, data, output, radii, lengths))
                        processes.append(p)
                    task_id+=1

                [x.start() for x in processes]
                [x.join() for x in processes]


            result_list = []
            result_list.append(output.keys())
            for elem in output:
                templist = []
                for key in output[elem].keys():
                    templist.append(output[elem][key])
                result_list.append(templist)

            return result_list

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)

    def __extract_select_training_worker(self, i, data, output, radii, lengths):

        try:
            combis = []
            for radius in radii:
                for length in lengths:
                    combi = [radius, length]
                    combis.append(combi)
            print("Motif extraction worker no: {0} length: {1}, radius: {2}".format(i, combis[i][1], combis[i][0]))
            X_indices, X_distances = self.extract_features(data=data,
                                              args=[combis[i][1], 16, combis[i][0], 'acceleration_abs'])
            X = self.select_features(data=data,
                                     args=[combis[i][1], 6, X_indices, 'acceleration_abs', X_distances])
            y = self.select_features(data=data,
                                     args=[combis[i][1], 1, X_indices, 'road_label', X_distances])

            print("Motif extraction worker no: {0} returned".format(i))
            output[i] = {
                'X': X,
                'y': y,
                'radius': combis[i][0],
                'length': combis[i][1],
                'motifs': 2
            }

            gc.collect()
        except Exception:
            self.logger.error(traceback.format_exc())
            gc.collect()
            #os._exit(2) Single workers should not crash the program

    @overrides
    def extract_select_inference_features(self, data, args=None, debug=False):
        """
        Extract-Select features
        :param data: pandas.DataFrame
        :param args:
        :return: list
        """

        try:
            X_train = args[2]
            length = args[0]
            radius=args[1]

            X_indices, X_distances = self.extract_features(data=data, args=[length,16,radius,'acceleration_abs'])
            #X_valid = self.select_features(data=data,
            #                         args=[length, 1, motifs, 'acceleration_abs'])
            X_valid = self.select_features(data=data,
                                                   args=[length, 6, X_indices, 'acceleration_abs', X_distances])


            if debug:
                y_valid = self.select_features(data=data,
                                                       args=[length, 1, X_indices, 'road_label', X_distances])

                return X_valid, y_valid

            return X_valid

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)