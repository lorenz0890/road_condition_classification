#https://www.sicara.ai/blog/2018-12-18-perfect-command-line-interfaces-python
#https://pypi.org/project/python-dotenv/
import click
import json
from pipeline.data_access.dao.sussex_huawei_dao import SussexHuaweiDAO
from pipeline.feature_engineering.preprocessing.sussex_huawei_preprocessor import SussexHuaweiPreprocessor
from pipeline.feature_engineering.feature_extraction.baseline_extractor import BaselineExtractor
from pipeline.feature_engineering.feature_extraction.mp_scrimp_extractor import MPScrimpExtractor
from pipeline.machine_learning.model.sklearn_model_factory import SklearnModelFactory
from pipeline.machine_learning.model.tslearn_model_factory import TslearnModelFactory
from sklearn.metrics import confusion_matrix
import pickle


@click.command()
@click.argument('config_path', nargs=-1)
@click.option('--training/--inference', '-t/-i')
def execute_command(config_path, training):
    # 1. Load config from path
    config = load_config(config_path[0])
    print(json.dumps(config, indent=1))

    # 2. Execute training or Inference
    if training:
        execute_training(config)
    else:
        execute_inference(config)

def init_pipeline(config):
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

def execute_training(config):
    # 1. Init pipeline
    print('--------------------INIT PIPELINE--------------------')
    dao, preprocessor, extractor, model_factory = init_pipeline(config)

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
    data_train, mean_train, std_train, data_valid = preprocessor.training_split_process(
        data=data,
        params=[labels, config['pre_proc_validation_sz'], config['pre_proc_training_sz'],
            config['data_set_column_names'][1:], config['pre_proc_movement_type_label'],
            config['pre_proc_road_type_label'], config['pre_proc_resample_freq']
        ]
    )

    #4. Feature extraction
    print('--------------------FEATURE EXTRACTION------------------')
    X_train = None
    if config['feature_eng_extractor_type'] == "motif":
        X_train = extractor.extract_select_training_features(
            data_train,
            [config['feature_eng_mp_extractor_radii'], config['feature_eng_mp_extractor_lengths']]
        )
    if config['feature_eng_extractor_type'] == "tsfresh":
        pass #TODO
    if X_train is None:
        pass#TODO Raise Error

    # 5. Find optimal classifier for given training set
    print('--------------------TRAINING PHASE----------------------')
    clf, score, conf, X_train, motif_len, motif_count = None, None, None, None, None, None
    if config['feature_eng_extractor_type'] == "motif":
        clf, score, conf, X_train, motif_len, motif_count = model_factory.find_optimal_model(
            'motif',
            X_train,
            config['classifier_optimal_search_space']
    )
    if config['feature_eng_extractor_type'] == "tsfresh":
        pass  # TODO
    if  clf is None or score is None or conf is None:
        pass#TODO Raise Error

    # 6. Prepare Validation
    print('--------------------PREPARE VALIDATION-------------------')
    X_valid, y_valid = None, None
    if config['feature_eng_extractor_type'] == "motif":
        X_valid, y_valid = extractor.extract_select_inference_features(
            data_valid, [X_train, motif_len], True
        )

    #7. Run Validation
    print('--------------------VALIDATION---------------------------')
    score = clf.score(X_valid, y_valid)
    print(score)
    y_pred = clf.predict(X_valid)
    conf = confusion_matrix(y_valid, y_pred, labels=None, sample_weight=None)
    print(conf)

    #8. Store Results
    print('--------------------STORE RESULTS------------------------')
    #TODO: delegate to DAO, make storing configureable
    with open('./X_train', 'wb') as X_train_file:
        pickle.dump(X_train, X_train_file)

    with open('./X_valid', 'wb') as X_valid_file:
        pickle.dump(X_valid, X_valid_file)
    with open('./y_valid', 'wb') as y_valid_file:
        pickle.dump(y_valid, y_valid_file)

    with open('./clf', 'wb') as clf_file:
        pickle.dump(clf, clf_file)

    meta_data ={
        'mean_train': mean_train,
        'std_train': std_train,
        'motif_len': motif_len,
        'motif_count':motif_count,
        'clf_score': score,
        'clf_conf': conf
    }
    with open('./meta_data', 'wb') as meta_file:
        pickle.dump(meta_data, meta_file)

def execute_inference(config):
    # 1. Init pipeline
    print('--------------------INIT PIPELINE--------------------')
    dao, preprocessor, extractor, model_factory = init_pipeline(config)

    # 2. Load data
    print('--------------------LOAD DATA------------------------')
    labels, data = dao.bulk_read_data(
        file_path=[config['data_set_path'], config['data_labels_path']],
        identifiers=config['data_set_trips'],
        column_names=[config['data_set_column_names'], config['data_label_column_names']],
        use_columns=[config['data_set_columns'], config['data_label_columns']]
    )
    #TODO: Delegate to DAO
    with open('./meta_data', 'rb') as meta_file:
        meta_data = pickle.load(meta_file)

    with open('./X_train', 'rb') as X_train_file:
        X_train = pickle.load(X_train_file)

    with open('./clf', 'rb') as clf_file:
        clf = pickle.load(clf_file)


    # 3. Preprocessing
    print('--------------------PRE PROCESSING--------------------')
    #TODO Use std, mean from training phase
    data_inference = preprocessor.inference_split_process(
        data=data,
        params=[
                config['data_set_column_names'][1:], config['pre_proc_movement_type_label'],
                config['pre_proc_resample_freq'], meta_data['mean_train'], meta_data['std_train']
                ]
    )

    # 4. Feature extraction
    print('--------------------FEATURE EXTRACTION------------------')
    X_inference = None
    if config['feature_eng_extractor_type'] == "motif":
        X_inference = extractor.extract_select_inference_features(data_inference,
                                              [
                                                  X_train,
                                                  meta_data['motif_len']
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
    # TODO Store best classifier, extracted features during training, preprocessing std, mean

def load_config(config_path):
    config = None
    with open(config_path) as json_file:
        config = json.load(json_file)

    return config #throw error of data is None

if __name__ == '__main__':
    execute_command()