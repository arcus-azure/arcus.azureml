from abc import ABCMeta, abstractmethod
import pandas as pd 
import numpy as np


class Trainer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def new_run(self, description: str = None) :
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_classifier(self, fitted_model, X_test: np.array, y_test: np.array, show_roc: bool = False, 
                            class_names: np.array = None, finish_existing_run: bool = True, upload_model: bool = True) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def get_best_model(self, metric_name:str, take_highest:bool = True):
        raise NotImplementedError
