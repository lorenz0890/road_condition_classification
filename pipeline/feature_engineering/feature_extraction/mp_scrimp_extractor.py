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

            return mtfs

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)


    @overrides
    def select_features(self, data, args=None):
        """
        Select features
        :param data: pandas.DataFrame
        :param args:
        :return: pandas.DataFrame
        """
        try:
            mtfs = args[2]
            sz = len([item for sublist in mtfs for item in sublist]) * args[0]
            attr_vec = numpy.ndarray(shape=(sz, args[1]), dtype=float) #3
            # print(mtfs)
            count = 0
            i = 1.0
            print(len(mtfs))
            print(sz)
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

            gc.collect()
        except Exception:
            self.logger.error(traceback.format_exc())
            gc.collect()
            #os._exit(2) Single workers should not crash the program


    def __extract_select_inference_worker(self, i, data, motif, motif_id, output, length):
        try:

            distances_valid = []
            motifs_valid = []
            motif_ids_valid = []

            print("Motif extraction worker no: {0} length: {1}".format(i, length))

            for j in range(0, len(data['acceleration_abs']) - length, 1):
                window = data['acceleration_abs'][j:j + length].values
                diff = None
                try:
                    diff = motif - window
                except ValueError:
                    print("Motif and windows len differ")
                    print(len(motif))
                    print(type(motif))
                    print(len(window))
                    print(type(window))

                except Exception as e:
                    self.logger.error(traceback.format_exc())
                    print(e)

                if not True in numpy.isnan(diff):
                    distances_valid.append(numpy.sqrt(numpy.sum(numpy.square(diff))))
                    motifs_valid.append(j)
                    motif_ids_valid.append(motif_id[0])

            print("Motif extraction worker no: {0} returned".format(i))
            output[i] = motifs_valid, motif_ids_valid, distances_valid

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

            #X_train = args[0]
            length = args[1]
            '''
            manager = mp.Manager()
            output = manager.dict()
            num_tasks = int(len(X_train) / length)
            task_id = 0
            num_processors = args[2]
            while task_id < num_tasks:
                processes = []
                for i in range(num_processors):
                    if num_processors >=0 and i < num_tasks: #TODO: Consider removal of first confition
                        split_sz = int(len(X_train) / length)
                        i = i * split_sz
                        motif = X_train[i:i + length][0].values
                        motif_id = X_train[i:i + length][1].values
                        p = mp.Process(target=self.__extract_select_inference_worker, args=(task_id, data, motif, motif_id,
                                                                                            output, length))
                        processes.append(p)
                    task_id += 1

                [x.start() for x in processes]
                [x.join() for x in processes]

            print('1')

            distances_valid_full = []
            motifs_valid_full = []
            motif_ids_valid_full = []
            for o in output:
                motifs_valid_full += output[o][0]
                motif_ids_valid_full += output[o][1]
                distances_valid_full += output[o][2]

            print('2')

            dm = list(zip(distances_valid_full, motifs_valid_full))
            dm.sort()
            m_sorted = [m for d, m in dm]

            dmid = list(zip(distances_valid_full, motifs_valid_full))
            dmid.sort()
            mid_sorted = [m for d, m in dmid]

            print('3')

            k = list(set(list(mid_sorted)))
            mtfs = []
            for i in range(len(k)): #2 is number of labels, make this configureable
                mtfs.append([])

            for i in range(int(len(m_sorted))):
                mtfs[k.index(mid_sorted[i])].append(m_sorted[i])
                #if int(mid_sorted[i]) == int(k[0]):
                #    mtfs[0].append(m_sorted[i])
                #else:
                #    mtfs[1].append(m_sorted[i])

            for i in range(len(k)): #2 is number of labels, make this configureable
                #import pdb; pdb.set_trace()
                mtfs[i] = [mtfs[i][0]]

            print('4')
            '''
            motifs, motif_d = self.extract_features(data, length)
            #X_valid = self.select_features(data=data,
            #                         args=[length, 1, motifs, 'acceleration_abs'])
            X_valid = self.select_features(data=data,
                                                   args=[length, 2, motifs, 'acceleration_abs'])


            if debug:
                y_valid = self.select_features(data=data,
                                                       args=[length, 1, motifs, 'road_label'])

                return X_valid, y_valid

            return X_valid

        except Exception:
            self.logger.error(traceback.format_exc())
            os._exit(2)