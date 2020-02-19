from pipeline.data_access.dao.sussex_huawei_dao import SussexHuaweiDAO
from pipeline.feature_engineering.preprocessing.sussex_huawei_preprocessor import SussexHuaweiPreprocessor
from pipeline.feature_engineering.feature_extraction.baseline_extractor import BaselineExtractor
from pipeline.feature_engineering.feature_extraction.mp_scrimp_extractor import MPScrimpExtractor
from pipeline.machine_learning.model.sklearn_model_factory import SklearnModelFactory
from pipeline.machine_learning.model.tslearn_model_factory import TslearnModelFactory
from pipeline.abstract_pipline_facade import PipelineFacade
from overrides import overrides
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import pandas
import numpy
from tsfresh.feature_extraction.settings import from_columns


class ConcretePipelineFacade(PipelineFacade):
    def __init__(self):
        super().__init__()

    @overrides
    def execute_training(self, config):
        """
        Run training based on config
        :param config: dict
        """

        # 1. Init pipeline
        print('--------------------INIT PIPELINE--------------------')
        dao, preprocessor, extractor, model_factory = self.init_pipeline(config)

        # 2. Load data
        print('--------------------LOAD DATA------------------------')
        labels, data = dao.bulk_read_data(
            file_path=[config['data_set_path'], config['data_labels_path']],
            identifiers=config['data_set_trips'],
            column_names=[config['data_set_column_names'], config['data_label_column_names']],
            use_columns=[config['data_set_columns'], config['data_label_columns']]
        )

        # 3. Preprocessing
        print('--------------------PRE PROCESSING--------------------')
        data_train, mean_train, std_train, data_test, data_valid = preprocessor.training_split_process(
            data=data,
            config=config,
            labels=labels
        )

        # 4. Feature extraction
        print('--------------------FEATURE EXTRACTION------------------')
        X_train = None
        X_test = None

        if config['feature_eng_extractor_type'] == "motif":
            X_train = extractor.extract_select_training_features(
                data_train,
                [config['feature_eng_mp_extractor_radii'], config['feature_eng_mp_extractor_lengths']]
            )
            X_test = extractor.extract_select_training_features(
                data_test,
                [config['feature_eng_mp_extractor_radii'], config['feature_eng_mp_extractor_lengths']]
            )

        segment_length = config['feature_eng_baseline_extractor_segement_len']
        if config['feature_eng_extractor_type'] == "ts-fresh":

            # TODO migrate the preperation for extraction to extract_select_training_features, make label column name configureable
            data_train = preprocessor.encode_categorical_features(data=data_train,
                                                                  mode='custom_function',
                                                                  columns=['road_label'],
                                                                  encoding_function=lambda x: (x > 2.0).astype(int)
                                                                  )  # 0 City, 1 Countryside

            data_test = preprocessor.encode_categorical_features(data=data_test,
                                                                  mode='custom_function',
                                                                  columns=['road_label'],
                                                                  encoding_function=lambda x: (x > 2.0).astype(int)
                                                                  )  # 0 City, 1 Countryside


            #Find segements with homogeneous labeling
            split = lambda df, chunk_size: numpy.array_split(df, len(df) // chunk_size + 1, axis=0)
            segments_train = split(data_train, segment_length)
            segments_test= split(data_test, segment_length)
            segments_train_homogeneous, segments_test_homogeneous = [], []
            for segment in segments_train:
                if segment.road_label.nunique() == 1 and segment.shape[0] == segment_length:
                    segments_train_homogeneous.append(segment)
            for segment in segments_test:
                if segment.road_label.nunique() == 1 and segment.shape[0] == segment_length:
                    segments_test_homogeneous.append(segment)

            data_train = pandas.concat(segments_train_homogeneous, axis=0)
            data_test = pandas.concat(segments_test_homogeneous, axis=0)


            #Generate id column
            train_id = [None]*data_train.index.size
            id = 0
            for i in range(0, data_train.index.size, segment_length):
                train_id[i:i+segment_length] = [id]*segment_length
                id+=1
            train_id = train_id[:data_train.index.size]
            data_train['id'] = train_id

            test_id = [None]*data_test.index.size
            id = 0
            for i in range(0, data_test.index.size, segment_length):
                test_id[i:i + segment_length] = [id]*segment_length
                id += 1
            test_id = test_id[:data_test.index.size]
            data_test['id'] = test_id


            y_train = data_train[['road_label', 'id']].reset_index(drop=True)
            y_train = y_train.groupby(y_train.index // segment_length).agg(lambda x: x.value_counts().index[0]) #majority label in segment
            X_train = data_train[['acceleration_abs', 'id']].reset_index(drop=True)
            y_test = data_test[['road_label', 'id']].reset_index(drop=True)
            y_test = y_test.groupby(y_test.index // segment_length).agg(lambda x: x.value_counts().index[0])
            X_test = data_test[['acceleration_abs', 'id']].reset_index(drop=True)

            print(y_train)

            #Extract Training features
            X_train = extractor.extract_select_training_features(
                X_train,
                args=['id', config['hw_num_processors'], None, y_train['road_label'], config['feature_eng_baseline_extractor_fdr']] #TODO use fdr dfrom cfg

            )

            #Get feature map for validation and training set
            kind_to_fc_parameters = from_columns(X_train)
            X_test = extractor.extract_select_inference_features(
                X_test,
                args=['id', config['hw_num_processors'], None, kind_to_fc_parameters]
            )

            X_train = ['placeholder',
                       [X_train, y_train['road_label'].rename(columns={'road_label': 0}, inplace=True),
                       'N/A', 'N/A', 'N/A']]  # required for further processing. TODO: Unifiy naming!

            X_test = ['placeholder',
                       [X_test, y_test['road_label'].rename(columns={'road_label': 0}, inplace=True),
                       'N/A', 'N/A', 'N/A']]

        if X_train is None or X_test is None:
            pass  # TODO Raise Error


        # 5. Find optimal classifier for given training set
        print('--------------------TRAINING PHASE----------------------')
        print(X_test[0][0])
        print(X_test[0][1])
        clf, score, conf, X_train, motif_len, motif_radius, motif_count = model_factory.find_optimal_model(
            config['feature_eng_extractor_type'],  # TODO remove bc deprecated. config handed anyways
            config,
            X_train,
            X_test,
        )
        if clf is None or score is None or conf is None:
            pass  # TODO Raise Error

        # 6. Prepare Validation
        print('--------------------PREPARE VALIDATION-------------------')

        X_valid, y_valid, kind_to_fc_parameters = None, None, None
        if config['feature_eng_extractor_type'] == "ts-fresh":

            data_valid = preprocessor.encode_categorical_features(data=data_valid,
                                                                 mode='custom_function',
                                                                 columns=['road_label'],
                                                                 encoding_function=lambda x: (x > 2.0).astype(int)
                                                                 )  # 0 City, 1 Countryside

            #Segement validation data ins pieces with homogeneous length
            split = lambda df, chunk_size: numpy.array_split(df, len(df) // chunk_size + 1, axis=0)
            segments_valid = split(data_valid, segment_length)
            segments_valid_homogeneous = []
            for segment in segments_valid:
                if segment.shape[0] == segment_length:
                    segments_valid_homogeneous.append(segment)

            data_valid = pandas.concat(segments_valid_homogeneous, axis=0)

            #Generate id column
            valid_id = [None] * data_valid.index.size
            id = 0
            for i in range(0, data_valid.index.size, segment_length):
                valid_id[i:i + segment_length] = [id] * segment_length
                id += 1
            valid_id = valid_id[:data_valid.index.size]
            data_valid['id'] = valid_id

            y_valid = data_valid[['road_label', 'id']].reset_index(drop=True)
            y_valid = y_valid.groupby(y_valid.index // segment_length).agg(lambda x: x.value_counts().index[0])
            X_valid = data_valid[['acceleration_abs', 'id']].reset_index(drop=True)

            # Get feature map for validation and training set
            kind_to_fc_parameters = from_columns(X_train)
            X_valid = extractor.extract_select_inference_features(
                X_valid,
                args=['id', config['hw_num_processors'], None, kind_to_fc_parameters]
            )

            y_valid = y_valid['road_label'].rename(columns={'road_label': 0}, inplace=True)

        if config['feature_eng_extractor_type'] == "motif":
            X_valid, y_valid = extractor.extract_select_inference_features(
                data_valid, [motif_radius, motif_len, config['hw_num_processors']], True
            )


        # 7. Run Validation
        print('--------------------VALIDATION---------------------------')
        if config['feature_eng_extractor_type'] == 'motif':
            print("Validation y label 1: {}".format(list(y_valid[0]).count(1.0) / len(y_valid)))  # TODO: make configureable
            print("Validation y label 3: {}".format(list(y_valid[0]).count(3.0) / len(y_valid)))
        elif config['feature_eng_extractor_type'] == 'ts-fresh':
            pass #TODO

        #print(X_valid)
        #print(y_valid)
        score = clf.score(X_valid, y_valid)
        print(score)
        y_pred = clf.predict(X_valid)
        conf = confusion_matrix(y_valid, y_pred, labels=None, sample_weight=None)
        print(conf)
        report = str(classification_report(y_valid, y_pred))
        best_params = clf.best_params_
        print(report)

        # 8. Store Results
        print('--------------------STORE RESULTS------------------------')
        # TODO: delegate to DAO, make storing configureable

        pandas.DataFrame(X_train).to_pickle('X_train.pkl')
        pandas.DataFrame(X_test).to_pickle('X_train.pkl')
        pandas.DataFrame(X_valid).to_pickle('X_valid.pkl')

        with open('./clf', 'wb') as clf_file:
            pickle.dump(clf, clf_file)

        meta_data = None
        if config['feature_eng_extractor_type'] == "motif":
            meta_data = {
                'mean_train': mean_train,
                'std_train': std_train,
                'motif_len': motif_len,
                'motif_radius' : motif_radius,
                'motif_count': motif_count,
                'clf_score': score,
                'clf_conf': conf,
                'clf_report': report,
                'clf_best_params': best_params
            }

        if config['feature_eng_extractor_type'] == "ts-fresh":
            meta_data = {
                'mean_train': mean_train,
                'std_train': std_train,
                'feature_mapping': kind_to_fc_parameters,
                'clf_score': score,
                'clf_conf': conf,
                'clf_report': report,
                'clf_best_params': best_params
            }


        with open('./meta_data', 'wb') as meta_file:
            pickle.dump(meta_data, meta_file)

    @overrides
    def execute_inference(self, config):
        """
        Run inference based on config
        :param config: dict
        """

        # 1. Init pipeline
        print('--------------------INIT PIPELINE--------------------')
        dao, preprocessor, extractor, model_factory = self.init_pipeline(config)

        # 2. Load data
        print('--------------------LOAD DATA------------------------')
        labels, data = dao.bulk_read_data(
            file_path=[config['data_set_path'], config['data_labels_path']],
            identifiers=config['data_set_trips'],
            column_names=[config['data_set_column_names'], config['data_label_column_names']],
            use_columns=[config['data_set_columns'], config['data_label_columns']]
        )
        # TODO: Delegate to DAO
        meta_data, X_train, clf = None, None, None
        with open('./meta_data', 'rb') as meta_file:
            meta_data = pickle.load(meta_file)

        with open('./clf', 'rb') as clf_file:
            clf = pickle.load(clf_file)

        # 3. Preprocessing
        print('--------------------PRE PROCESSING--------------------')
        params = []
        params += config['data_set_column_names'][1:]
        params.append(config['pre_proc_resample_freq'])
        params.append(meta_data['mean_train'])
        params.append(meta_data['std_train'])

        data_inference = preprocessor.inference_split_process(
            data=data,
            params=params
        )

        # 4. Feature extraction
        print('--------------------FEATURE EXTRACTION------------------')
        X_inference = None
        if config['feature_eng_extractor_type'] == "motif":
            X_inference = extractor.extract_select_inference_features(data_inference,
                                                                      [
                                                                          meta_data['motif_len'],
                                                                          meta_data['motif_radius'],
                                                                          config['hw_num_processors']
                                                                      ], False
                                                                      )
        if config['feature_eng_extractor_type'] == "ts-fresh":

            valid_id = [None] * data_inference.index.size
            id = 0
            for i in range(0, data_inference.index.size, 30):
                valid_id[i:i + 30] = [id] * 30
                id += 1
            valid_id = valid_id[:data_inference.index.size]
            data_inference['id'] = valid_id

            X_valid = data_inference[['acceleration_abs', 'id']].reset_index(drop=True)

            # Get feature map for validation and training set
            kind_to_fc_parameters = meta_data['feature_mapping']
            X_valid = extractor.extract_select_inference_features(
                X_valid,
                args=['id', config['hw_num_processors'], None, kind_to_fc_parameters]
            )

        if X_inference is None:
            pass  # TODO Raise Error

        # 5. Inference
        print('--------------------INFERENCE PHASE----------------------')
        y_pred = clf.predict(X_inference)

        # 6. Store Results
        print('--------------------STORE RESULTS------------------------')
        print(y_pred)
        pandas.DataFrame(y_pred).to_pickle('y_pred.pkl')


    @overrides
    def init_pipeline(self, config):
        """
        Initialize pipline based on config
        :param config: dict
        """

        dao = None
        preprocessor = None
        extractor = None
        model_factory = None
        if config['data_set_type'] == "sussex_huawei":
            dao = SussexHuaweiDAO()
            preprocessor = SussexHuaweiPreprocessor()
        if config['feature_eng_extractor_type'] == "motif":
            extractor = MPScrimpExtractor()
        if config['feature_eng_extractor_type'] == "ts-fresh":
            extractor = BaselineExtractor()
        if config['classifier_model_factory_type'] == "sklearn":
            model_factory = SklearnModelFactory()
        if config['classifier_model_factory_type'] == "tslearn":
            model_factory = TslearnModelFactory()

        if dao is None or preprocessor is None or extractor is None or model_factory is None:
            # TODO: Raise appropriate error
            pass
        return dao, preprocessor, extractor, model_factory