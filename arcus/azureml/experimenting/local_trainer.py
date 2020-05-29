from arcus.azureml.experimenting import trainer
from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
from azureml.data.datapath import DataPath
from azureml.data.dataset_error_handling import DatasetValidationError, DatasetExecutionError
from azureml.data.dataset_type_definitions import PromoteHeadersBehavior
from datetime import datetime
import sklearn.metrics as metrics
import joblib
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from itertools import product, combinations
from logging import Logger
import logging
import sys

class LocalMLTrainer(trainer.Trainer):
    is_connected: bool = False
    __logger: Logger = None

    def new_run(self, description: str = None, copy_folder: bool = False, metrics: dict = None) :
        raise NotImplementedError
    
    def complete_run(self, fitted_model, metrics: dict = None, upload_model: bool = True):
        raise NotImplementedError
    
    def evaluate_classifier(self, fitted_model, X_test: np.array, y_test: np.array, show_roc: bool = False, 
                            class_names: np.array = None, finish_existing_run: bool = True, upload_model: bool = True, return_predictions: bool = False) -> np.array:
        raise NotImplementedError

    def get_best_model(self, metric_name:str, take_highest:bool = True):
        raise NotImplementedError

    def add_tuning_result(self, run_index: int, train_score: float, test_score: float, sample_count: int, durations:np.array, parameters: dict, estimator):
        raise NotImplementedError

    def grid_search(self, model, hyper_parameters: dict, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, constructor_parameters: dict = None, validation_method = None, use_aml_compute: bool = False, take_highest:bool = True):
        raise NotImplementedError