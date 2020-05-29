from abc import ABCMeta, abstractmethod
import pandas as pd 
import numpy as np


class Trainer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def new_run(self, description: str = None, copy_folder: bool = False, metrics: dict = None) :
        raise NotImplementedError
    
    @abstractmethod
    def complete_run(self, fitted_model, metrics: dict = None, upload_model: bool = True):
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_classifier(self, fitted_model, X_test: np.array, y_test: np.array, show_roc: bool = False, 
                            class_names: np.array = None, finish_existing_run: bool = True, upload_model: bool = True, return_predictions: bool = False) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def get_best_model(self, metric_name:str, take_highest:bool = True):
        raise NotImplementedError

    @abstractmethod
    def add_tuning_result(self, run_index: int, train_score: float, test_score: float, sample_count: int, durations:np.array, parameters: dict, estimator):
        raise NotImplementedError

    @abstractmethod
    def grid_search(self, model, hyper_parameters: dict, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, constructor_parameters: dict = None, validation_method = None, use_aml_compute: bool = False, take_highest:bool = True):
        raise NotImplementedError
