import os
import sys
from arcus.azureml.environment.environment_factory import WorkEnvironmentFactory as fac
from arcus.azureml.environment.aml_environment import AzureMLEnvironment as aml

from collections import Counter
# Imports for the excercise
import pandas as pd 
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pytest
from datetime import datetime

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization,SpatialDropout1D,Bidirectional, Embedding, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from arcus.azureml.experimenting.tuning import LocalArcusGridSearchCV

def is_interactive():
    # If the environment variable System_DefinitionId is not available, we run locally
    return 'SYSTEM_DEFINITIONID' not in os.environ

def setup_function(func):
    if not is_interactive():
        import pytest
        pytest.skip('Test only runs when interactive mode enable')




datastore_name = 'smart_devops'
partitioned_datastore_name = 'aiincubators_covid'

def test_factory_dataset():
    work_env = fac.Create(connected=True)
    _df = work_env.load_tabular_dataset('smart-devops-changesets')
    _df = _df.tail(20)
    assert _df.shape == (20,16) # 16 columns expected


def test_dataset():
    work_env = aml.Create()
    _df = work_env.load_tabular_dataset('smart-devops-changesets')
    _df = _df.tail(20)
    assert _df.shape == (20,16) # 16 columns expected

def test_partitions():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition(partition_name= 'test-partitioning/stock_AT*', columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
    assert partition_df is not None
    assert partition_df.shape == (30, 9)

    assert len(Counter(partition_df['Isin'])) == 3

def test_partitions_header():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition('test-partitioning/stock_header_AT*', first_row_header=True)
    assert 'Isin' in partition_df.columns
    assert partition_df.shape == (24, 9)
    assert len(Counter(partition_df['Isin'])) == 3


def test_partitions_noheader_and_columns():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition('test-partitioning/stock_header_AT*', first_row_header=False, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
    assert partition_df is not None

    assert 'C' in partition_df.columns
    assert partition_df.shape == (27, 9)

def test_partitions_header_and_columns():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition('test-partitioning/stock_header_AT*', first_row_header=True, columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
    assert partition_df is not None

    assert 'Isin' in partition_df.columns
    assert partition_df.shape == (24, 9)
    assert len(Counter(partition_df['Isin'])) == 3

def test_partitions_notexisting():
    work_env = fac.Create(connected=True, datastore_path='arcus_partition_test')
    partition_df = work_env.load_tabular_partition('test-partitioning/stock_BE*', columns=['Close', 'High', 'Isin', 'ItemDate', 'Low', 'Market', 'Open', 'Ticker', 'Volume'])
    assert partition_df is None

def test_classification_evaluation_keras():
    if not is_interactive():
        pytest.skip('Test only runs when interactive mode enable')

    df = pd.read_csv('tests/resources/datasets/student-admission.csv')
    y = df.pop('Admission')
    X = np.asarray(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    y_train = np_utils.to_categorical(y_train)

    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=2, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    amlenv = fac.Create(connected = True, config_file='.azureml/config.json')
    trainer = amlenv.start_experiment('arcus-unit-tests')
    _run = trainer.new_run('neuralnet')
    model.fit(X_train, y_train)
    
    trainer.evaluate_classifier(model, X_test, y_test, show_roc = True, upload_model = True)   

def test_classification_evaluation_sklearn():
    if not is_interactive():
        pytest.skip('Test only runs when interactive mode enable')

    amlenv = fac.Create(connected = True, config_file='.azureml/config.json')
    trainer = amlenv.start_experiment('arcus-unit-tests')
    _run = trainer.new_run('logreg-'+ datetime.now().strftime("%Y%m%d-%H%M%S"))
    df = pd.read_csv('tests/resources/datasets/student-admission.csv')
    y = df.Admission.values
    X = np.asarray(df.drop(['Admission'],axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    df.tail(5)
    logreg = linear_model.LogisticRegression(solver='liblinear', C=1.1)
    logreg.fit(X_train, y_train)
    
    trainer.evaluate_classifier(logreg, X_test, y_test, show_roc = True, upload_model = True)   

def build_sequential_model(neurons: int = 10):
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=2, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def test_classification_evaluation_keras_gridsearch():
    if not is_interactive():
        pytest.skip('Test only runs when interactive mode enable')

    df = pd.read_csv('tests/resources/datasets/student-admission.csv')
    y = df.pop('Admission')
    X = np.asarray(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    y_train = np_utils.to_categorical(y_train)


    amlenv = fac.Create(connected = True, config_file='.azureml/config.json')
    trainer = amlenv.start_experiment('arcus-unit-tests')
    _run = trainer.new_run('neuralnet')
    model = KerasClassifier(build_fn=build_sequential_model, batch_size=16, epochs =4)
    param_grid = {'neurons':[5]}
    grid = LocalArcusGridSearchCV(estimator=model, param_grid = param_grid,verbose=3, active_trainer=trainer)

    fitted_grid = grid.fit(X_train, y_train) 
    best_classifier = fitted_grid.best_estimator_.model
    trainer.evaluate_classifier(best_classifier, X_test, y_test, show_roc = True, upload_model = True)   

def test_logging_dictionary():
    if not is_interactive():
        pytest.skip('Test only runs when interactive mode enable')

    amlenv = fac.Create(connected = True, config_file='.azureml/config.json')
    trainer = amlenv.start_experiment('arcus-unit-tests')
    _run = trainer.new_run('logging dict', metrics={'BatchSize': 30, 'Epochs': 40})

    _run.complete()

def test_secret_existing():
    secret_key_name = 'unittestsecret'
    if not is_interactive():
        pytest.skip('Test only runs when interactive mode enable')

    amlenv = fac.Create(connected = True, config_file='.azureml/config.json')
    secret_value = amlenv.get_secret(secret_key_name)

    assert(secret_value == 'unittestvalue')

def test_secret_notexisting():
    secret_key_name = 'unittestsecret2'
    if not is_interactive():
        pytest.skip('Test only runs when interactive mode enable')

    amlenv = fac.Create(connected = True, config_file='.azureml/config.json')
    secret_value = amlenv.get_secret(secret_key_name)

    assert(secret_value == None)