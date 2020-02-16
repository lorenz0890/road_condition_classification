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
        #import random  # for debugging
        labels, data = dao.bulk_read_data(
            file_path=[config['data_set_path'], config['data_labels_path']],
            identifiers=config['data_set_trips'],
            # identifiers=random.sample(config['data_set_trips'], len(config['data_set_trips'])//4),
            column_names=[config['data_set_column_names'], config['data_label_column_names']],
            use_columns=[config['data_set_columns'], config['data_label_columns']]
        )

        # 3. Preprocessing
        print('--------------------PRE PROCESSING--------------------')
        #params = [labels, config['pre_proc_test_sz'], config['pre_proc_training_sz'], config['pre_proc_validation_sz']]
        #params += config['data_set_column_names'][1:] + [config['pre_proc_movement_type_label']]
        #params += [config['pre_proc_road_type_label']]
        #params.append(config['pre_proc_resample_freq'])

        print(data.head(10))
        data_train, mean_train, std_train, data_test, data_valid = preprocessor.training_split_process(
            data=data,
            config=config,
            labels=labels
        )

        print(data_train.head(10))
        # 4. Feature extraction
        print('--------------------FEATURE EXTRACTION------------------')
        X_train = None
        X_test = None
        if config['feature_eng_extractor_type'] == "motif":
            #data_train.drop(['id'], axis=1, inplace=True)
            #data_test.drop(['id'], axis=1, inplace=True)
            #data_valid.drop(['id'], axis=1, inplace=True)
            X_train = extractor.extract_select_training_features(
                data_train,
                [config['feature_eng_mp_extractor_radii'], config['feature_eng_mp_extractor_lengths']]
            )
            X_test = extractor.extract_select_training_features(
                data_test,
                [config['feature_eng_mp_extractor_radii'], config['feature_eng_mp_extractor_lengths']]
            )
        if config['feature_eng_extractor_type'] == "tsfresh":
            """
            # TODO migrate the preperation for extraction to extract_select_training_features, make label column name configureable
            data_train = preprocessor.encode_categorical_features(data=data_train,
                                                                  mode='custom_function',
                                                                  columns=['road_label'],
                                                                  encoding_function=lambda x: (x > 2.0).astype(int)
                                                                  )  # 0 City, 1 Countryside
            y_train = data[['road_label']].reset_index(drop=True)
            data['id'] = range(1, len(data) + 1) #what happens if i just set this to 1
            y_train['id'] = data['id']
            y_train['road_label'].index = list(y_train['id'])

            X_train = extractor.extract_select_training_features(
                data_train,
                args=['id', config['hw_num_processors'], None, y_train['road_label'], 0.1]

            )

            keys = X_train.keys()
            keys = list(filter(lambda x: "acceleration_abs" in x, keys))

            X_join = pandas.concat([X_train, y_train], axis=1)
            X_join = preprocessor.remove_nans(X_join, replacement_mode='del_row')
            X_join[['road_label']] = X_join[['road_label']].astype('int')
            X_segments = preprocessor.segment_data(X_join, mode='labels',
                                                   label_column='road_label',
                                                   args=[0, 1])

            segment_length = 30  # 60s best in paper, 90 best in my evaluation, tested 30, 60, 90, 120
            X_segments_new = []
            for ind in range(0, len(X_segments)):
                X_segments_new = X_segments_new + preprocessor.segment_data(
                    X_segments[ind],
                    mode='fixed_interval',
                    args=[segment_length, True, True]
                )

            print(len(X_segments_new))
            keys.append('road_label')
            X_combined = preprocessor.de_segment_data(X_segments_new, keys)
            X_train, y_train = X_combined[keys[:-1]], X_combined[keys[-1]]
            X_train = ['placeholder',
                       [X_train, y_train, 'N/A', 'N/A', 'N/A']]  # required for further processing. TODO: Unifiy naming!
            """
            pass
        if X_train is None or X_test is None:
            pass  # TODO Raise Error

        #print(X_train)
        print(X_train[1][0].head(10))
        print(X_train[1][1].head(10))
        print(X_test[1][0].head(10))
        print(X_test[1][1].head(10))

        # 5. Find optimal classifier for given training set
        print('--------------------TRAINING PHASE----------------------')
        clf, score, conf, X_train, motif_len, motif_radius, motif_count = model_factory.find_optimal_model(
            'motif',  # TODO remove bc deprecated. X_train now decides mode(?)
            config,
            X_train,
            X_test,
        )
        if clf is None or score is None or conf is None:
            pass  # TODO Raise Error

        # 6. Prepare Validation
        print('--------------------PREPARE VALIDATION-------------------')
        #TODO: Adapat for TS Fresh
        print(data_valid.shape)
        print(data_valid.head(10))
        X_valid, y_valid = None, None
        #params = []
        #params += config['data_set_column_names'][1:]
        #params.append(config['pre_proc_resample_freq'])
        #params.append(mean_train)
        #params.append(std_train)

        #data_inference= preprocessor.inference_split_process(
        #    data=data_valid,
        #    config=config,
        #    meta_data={'mean_train': mean_train, 'std_train': std_train}
        #)
        if config['feature_eng_extractor_type'] == "motif":
            X_valid = extractor.extract_select_training_features(
                data_valid, [[motif_radius], [motif_len], config['hw_num_processors']]
            )

        y_valid = X_valid[1][1]
        X_valid =  X_valid[1][0]

        print(X_valid.head(10))
        print(y_valid.head(10))


        # 7. Run Validation
        print('--------------------VALIDATION---------------------------')
        print(X_valid.shape)
        score = clf.score(X_test, y_valid)
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

        X_train.to_pickle('X_train.pkl')
        X_valid.to_pickle('X_valid.pkl')

        with open('./clf', 'wb') as clf_file:
            pickle.dump(clf, clf_file)

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

        with open('./X_train', 'rb') as X_train_file:
            X_train = pickle.load(X_train_file)

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
        if config['feature_eng_extractor_type'] == "tsfresh":
            pass  # TODO
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
        if config['feature_eng_extractor_type'] == "tsfresh":
            extractor = BaselineExtractor()
        if config['classifier_model_factory_type'] == "sklearn":
            model_factory = SklearnModelFactory()
        if config['classifier_model_factory_type'] == "tslearn":
            model_factory = TslearnModelFactory()

        if dao is None or preprocessor is None or extractor is None or model_factory is None:
            # TODO: Raise appropriate error
            pass
        return dao, preprocessor, extractor, model_factory