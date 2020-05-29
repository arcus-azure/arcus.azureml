from abc import ABCMeta, abstractmethod
import pandas as pd 
import numpy as np
import sklearn.metrics as metrics
import joblib
import os
import matplotlib.pyplot as plt


class Trainer:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def new_run(self, description: str = None, copy_folder: bool = False, metrics: dict = None) :
        raise NotImplementedError
    
    @abstractmethod
    def _complete_run(self):
        raise NotImplementedError

    @abstractmethod
    def get_best_model(self, metric_name:str, take_highest:bool = True):
        raise NotImplementedError

    @abstractmethod
    def add_tuning_result(self, run_index: int, train_score: float, test_score: float, sample_count: int, durations:np.array, parameters: dict, estimator):
        raise NotImplementedError

    @abstractmethod
    def _log_confmatrix(self, confusion_matrix: np.array, class_names: np.array):
        raise NotImplementedError

    @abstractmethod
    def _log_metrics(self, metric_name: str, metric_value: float, description:str = None):
        raise NotImplementedError

    @abstractmethod
    def _save_roc_curve(self, roc_auc: float, roc_plot: plt):
        raise NotImplementedError

    
    def complete_run(self, fitted_model, metrics_to_log: dict = None, upload_model: bool = True):
        '''
        Saves all results of the active Run and completes it
        Args:
            fitted_model (model): The already fitted model to be tested.  Sklearn and Keras models have been tested
            metrics_to_log (dict): The metrics that should be logged with the model to the run
            upload_model (bool): This will upload the model (pkl file or json) to AzureML run (defaults to True)
        '''
        is_keras = 'keras' in str(type(fitted_model))

        if(metrics_to_log is not None):
            for k, v in metrics_to_log.items():
                self._log_metrics(k, v)
        
        if upload_model:
            # Save the model to the outputs directory for capture
            if(is_keras):
                model_folder_name = 'outputs/model'
                fitted_model.save(model_folder_name)
                files_to_upload = dict()
            else:
                model_file_name = 'outputs/model.pkl'
                joblib.dump(value = fitted_model, filename = model_file_name)

        self._complete_run()


    def __log_roc_curve(self, y_pred: np.array, y_test: np.array):
        '''Will upload the Receiver Operating Characteristic (ROC) Curve for binary classifiers

        Args:
            y_pred (np.array): The predicted values of the test set 
            y_test (np.array): The actual outputs of the test set

        Returns: 
            float: The ROC_AUC value
        '''
        # calculate the fpr and tpr for all thresholds of the classification
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.cla()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        self._save_roc_curve(roc_auc, plt)
        plt.show(block=False)
        plt.close()
        return roc_auc

    def evaluate_classifier(self, fitted_model, X_test: np.array, y_test: np.array, show_roc: bool = False, 
                             class_names: np.array = None, finish_existing_run: bool = True, upload_model: bool = True, return_predictions: bool = False) -> np.array:

        '''
        Will predict and evaluate a model against a test set and save all results to the active Run on AzureML
        Args:
            fitted_model (model): The already fitted model to be tested.  Sklearn and Keras models have been tested
            X_test (np.array): The test set to calculate the predictions with
            y_test (np.array): The output test set to evaluate the predictions against
            show_roc (bool): This will upload the ROC curve to the run in case of a binary classifier
            class_names (np.array): The class names that will be linked to the Confusion Matrix.  If not provided, the unique values of the y_test matrix will be used
            finish_existing_run (bool): Will complete the existing run on AzureML (defaults to True)
            upload_model (bool): This will upload the model (pkl file) to AzureML run (defaults to True)
            return_predictions (bool): If true, the y_pred values will be returned
        Returns: 
            np.array: The predicted (y_pred) values against the model
        '''
        is_keras = 'keras' in str(type(fitted_model))
        
        # Predict X_test with model
        if(is_keras):
            if 'predict_classes' in dir(fitted_model):
                y_pred = fitted_model.predict_classes(X_test)
            else:
                y_pred = fitted_model.predict(X_test)
                y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = fitted_model.predict(X_test)

        if class_names is None:
            class_names = np.char.mod('%d', sorted(np.unique(y_test)))

        # Print classification report
        print(metrics.classification_report(y_test, y_pred))

        # Confusion matrix
        cf = metrics.confusion_matrix(y_test, y_pred)
        self._log_confmatrix(cf, class_names)

        # Accuracy
        accuracy = metrics.accuracy_score(y_test, y_pred) * 100
        self._log_metrics('accuracy', accuracy, description='')

        if(show_roc == True):
            # Verify that we are having a binary classifier
            if(len(class_names)!=2):
                raise AttributeError('Showing a ROC curve is only possible for binary classifier, not for multi class')
            self.__log_roc_curve(y_test, y_pred) 

        if (finish_existing_run):
            self.complete_run(fitted_model, upload_model = upload_model)

        if return_predictions:  
            return y_pred
