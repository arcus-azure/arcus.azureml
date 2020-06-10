import os
from arcus.azureml.experimenting.tuning import LocalArcusGridSearchCV
# Imports for the excercise
import pandas as pd 
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pytest
import arcus.azureml.environment.aml_environment as aml
from datetime import datetime
import io

def is_interactive():
    # If the environment variable System_DefinitionId is not available, we run locally
    return 'SYSTEM_DEFINITIONID' not in os.environ

def test_local_gridsearch_aml_logging():
    if not is_interactive():
        pytest.skip('Test only runs when interactive mode enable')

    amlenv = aml.AzureMLEnvironment.Create(config_file='.azureml/config.json')
    trainer = amlenv.start_experiment('arcus-unit-tests')

    _run = trainer.new_run('logreg-'+ datetime.now().strftime("%Y%m%d-%H%M%S"))
    df = pd.read_csv('tests/resources/datasets/student-admission.csv')
    y = df.Admission.values
    X = np.asarray(df.drop(['Admission'],axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    df.tail(5)
    logreg = linear_model.LogisticRegression(solver='liblinear')
    param_grid = {'C':[0.1,10]}
    grid = LocalArcusGridSearchCV(logreg, param_grid, scoring='accuracy', active_trainer=trainer)
    grid.fit(X_train, y_train)

def test_setup_training():
    if not is_interactive():
        pytest.skip('Test only runs when interactive mode enable')

    amlenv = aml.AzureMLEnvironment.Create(config_file='.azureml/config.json')
    trainer = amlenv.start_experiment('arcus-unit-tests')
    trainer.setup_training('unit-test')
    assert os.path.exists(os.path.join('unit-test', 'train.py'))
