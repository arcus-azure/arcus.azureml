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
from arcus.azureml.experimenting.tuning import LocalArcusGridSearchCV
from arcus.azureml.environment.environment import WorkEnvironment

class LocalMLTrainer(trainer.Trainer):
    is_connected: bool = False
    __logger: Logger = None
    __data_directory: str = 'outputs'
    __environment: WorkEnvironment = None
    __current_experiment_name: str

    def __init__(self, experiment_name: str, loc_workspace: WorkEnvironment):
        '''
        Initializes a new disconnected Trainer that will persist and log all runs on the local workspace
        Args:
            experiment_name (str): The name of the experiment that will be used as reference
            loc_workspace (WorkEnvironment): The local WorkEnvironment
        '''
        self.__environment = loc_workspace
        self.__current_experiment_name = experiment_name
        self.__logger = logging.getLogger()

    def new_run(self, description: str = None, copy_folder: bool = False, metrics: dict = None) :
        '''
        This will begin a new local run.  When a previous run was still active, it will be completed.
        Args:
            description (str): An optional description that will be added to the run metadata
        '''
        self.__logger.info("Starting new run", description, " with following metrics: ", metrics)
        

    def get_best_model(self, metric_name:str, take_highest:bool = True):
        raise NotImplementedError

    def add_tuning_result(self, run_index: int, train_score: float, test_score: float, sample_count: int, durations:np.array, parameters: dict, estimator):
        self.__logger.info('Tuning score: ', train_score, ' - Test score: ', test_score)
        
    def _log_metrics(self, metric_name: str, metric_value: float, description:str = None):
        print(metric_name, metric_value) 
    
    def _complete_run(self):
        self.__logger.info("Completing run with following metrics: ", metrics)


    def _log_confmatrix(self, confusion_matrix: np.array, class_names: np.array):
        print(confusion_matrix)

    def _save_roc_curve(self, roc_auc: float, roc_plot: plt):
        self._log_metrics('roc_auc', roc_auc)