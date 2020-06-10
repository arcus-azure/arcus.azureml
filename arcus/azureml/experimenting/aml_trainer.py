from arcus.azureml.experimenting import trainer
from arcus.azureml.experimenting import errors

from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.data.datapath import DataPath
from azureml.data.dataset_error_handling import DatasetValidationError, DatasetExecutionError
from azureml.data.dataset_type_definitions import PromoteHeadersBehavior
from azureml.train.estimator import Estimator
from azureml.widgets import RunDetails

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
import os
import os.path
import shutil

class AzureMLTrainer(trainer.Trainer):
    is_connected: bool = False
    __config_file: str = '.azureml/config.json'
    __workspace: Workspace = None
    __experiment: Experiment = None
    __current_experiment_name: str
    __current_run: Run = None
    __logger: Logger = None
    __vm_size_list: list = None

    def __init__(self, experiment_name: str, aml_workspace: Workspace, aml_run: Run = None):
        '''
        Initializes a new connected Trainer that will persist and log all runs on AzureML workspace
        Args:
            experiment_name (str): The name of the experiment that will be seen on AzureML
            aml_workspace (Workspace): The connected workspace on AzureML
        '''
        self.__workspace = aml_workspace
        self.__logger = logging.getLogger()
        if aml_run is not None:
            self.__current_run = aml_run
            self.__experiment = aml_run.experiment
            self.__current_experiment_name = aml_run.experiment.name
        else:
            self.__current_experiment_name = experiment_name
            self.__experiment = Experiment(workspace=self.__workspace, name=experiment_name)


    @classmethod
    def CreateFromContext(cls):
        '''
        Creates a Trainer, based on the current Run context.  This will only work when used in an Estimator
        Returns: 
            AzureMLTrainer: an instance of AzureMLTrainer allowing the user to work connected.
        '''   
        run = Run.get_context()
        return cls(run.experiment.name, run.experiment.workspace, run)


    def new_run(self, description: str = None, copy_folder: bool = True, metrics: dict = None) -> Run:
        '''
        This will begin a new interactive run on the existing AzureML Experiment.  When a previous run was still active, it will be completed.
        Args:
            description (str): An optional description that will be added to the run metadata
            copy_folder (bool): Indicates if the output folder should be snapshotted and persisted
            metrics (dict): The metrics that should be logged in the run already
        Returns:
            Run: the AzureML Run object that can be used for further access and custom logic
        '''
        if(self.__current_run is not None):
            self.__current_run.complete()
        if(copy_folder):
            self.__current_run = self.__experiment.start_logging()
        else:
            self.__current_run = self.__experiment.start_logging(snapshot_directory = None)

        if(metrics is not None):
            for k, v in metrics.items():
                self.__current_run.log(k, v)

        if(description is not None):
            self.__current_run.log('Description', description)
        
        return self.__current_run

    def add_tuning_result(self, run_index: int, train_score: float, test_score: float, sample_count: int, durations:np.array, parameters: dict, estimator):
        '''
        This add results of a cross validation fold to the child run in a Grid Search
        Args:
            train_score (float): The given score of the training data
            test_score (float): The given score of the test data
            sample_count (int): The number of samples that were part of a fold
            durations (np.array): The different durations of the Grid Search
            parameters (dict): The parameter combinations that have been tested in this cross validation fold
            estimate (model): The actual fitted estimator / model that was trained in this fold
        '''
        _child_run = self.__current_run.child_run('Gridsearch' + str(run_index))
        self.__current_run.log_row('Trainscore', score = train_score)
        self.__current_run.log_row('Testscore', score = test_score)

        _table = {
            'Testing score': test_score,
            'Training score': train_score
            }

        for k in parameters.keys():
            v = parameters[k]
            if(v is None):
                v = 'None'
            _child_run.log(k, v)
            _table[k] = v
        
        self.__current_run.log_row('Results', '', **_table)
        _child_run.complete()

    def get_best_model(self, metric_name:str, take_highest:bool = True):
        '''
        Tags and returns the best model of the experiment, based on the given metric
        Args:
            metric_name (str): The name of the metric, such as accuracy
            take_highest (bool): In case of accuracy and score, this is typically True.  In case you want to get the model based on the lowest error, you can use False
        Returns:
            Run: the best run, which will be labeled as best run
        '''
        runs = {}
        run_metrics = {}
        for r in tqdm(self.__experiment.get_runs()):
            metrics = r.get_metrics()
            if metric_name in metrics.keys():
                runs[r.id] = r
                run_metrics[r.id] = metrics
        best_run_id = min(run_metrics, key = lambda k: run_metrics[k][metric_name])
        best_run = runs[best_run_id]
        best_run.tag('Best run')
        return best_run

    def get_azureml_experiment(self):
        '''
        Gives access to the AzureML experiment object
        Returns:
            Experiment: the existing experiment
        '''
        return self.__experiment
        
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


    def setup_training(self, training_name: str, overwrite: bool = False):
        '''
        Will initialize a new directory (using the given training_name) and add a training script and requirements file to run training
        Args:
            training_name (str): The name of a training.  This will be used to create a directory.  Can contain subdirectory
            overwrite (bool): Defines if the existing training files should be overwritten
        '''
        if not os.path.exists(training_name):
            os.makedirs(training_name)
        # Take default training script and copy to the new folder
        default_training_script_file = os.path.join(str(os.path.dirname(__file__)), 'resources/train.py')
        default_requirements_file = os.path.join(str(os.path.dirname(__file__)), 'resources/requirements.txt')

        if overwrite or os.path.isfile(default_training_script_file):
            shutil.copy2(default_training_script_file, training_name)

        if overwrite or os.path.isfile(default_requirements_file):
            shutil.copy2(default_requirements_file, training_name)
        


    def start_training(self, training_name: str, estimator_type: str = None, input_datasets: np.array = None, input_datasets_to_download: np.array = None, compute_target:str='local', gpu_compute: bool = False, script_parameters: dict = None, show_widget: bool = True, **kwargs):
        ''' 
        Will start a new training, taking the training name as the folder of the run
        Args:
            training_name (str): The name of a training.  This will be used to create a directory.  Can contain subdirectory
            input_datasets (np.array): An array of data set names that will be mounted on the compute in a directory of the dataset name
            input_datasets_to_download (np.array): An array of data set names that will be downloaded to the compute in a directory of the dataset name
            compute_target (str): The compute target (default = 'local') on which the training should be executed
            gpu_compute (bool): Indicates if GPU compute is required for this script or not
            script_parameters (dict): A dictionary of key/value parameters that will be passed as arguments to the training script
        '''
        # Check if directory exists
        if not(os.path.exists(training_name) and os.path.isdir(training_name)):
            raise FileNotFoundError(training_name)

        # Check compute target
        if compute_target != 'local':
            self.__check_compute_target(compute_target, gpu_compute)
            

        # Add datasets
        datasets = list()
        if(input_datasets is not None):
            for ds in input_datasets:
                datasets.append(self.__workspace.datasets[ds].as_named_input(ds).as_mount(path_on_compute=ds))
        if(input_datasets_to_download is not None):
            for ds in input_datasets_to_download:
                datasets.append(self.__workspace.datasets[ds].as_named_input(ds).as_download(path_on_compute=ds))

        # as mount - as download
        constructor_parameters = {
            'source_directory':training_name,
            'script_params':script_parameters,
            'inputs':datasets,
            'compute_target':compute_target,
            'entry_script':'train.py',
            'pip_requirements_file':'requirements.txt', 
            'use_gpu':gpu_compute,
            'use_docker':True}
        
        print('Creating estimator of type', estimator_type)

        if(estimator_type is None):
            # Using default Estimator
            estimator = Estimator(**constructor_parameters)
        elif(estimator_type == 'tensorflow'):
            from azureml.train.dnn import TensorFlow
            version_par = 'framework_version'
            if(not version_par in constructor_parameters.keys()):
                print('Defaulting to version 2.0 for TensorFlow')
                constructor_parameters[version_par] = '2.0'
            estimator = TensorFlow(**constructor_parameters)
        elif(estimator_type == 'sklearn'):
            from azureml.train.sklearn import SKLearn
            estimator = SKLearn(**constructor_parameters)
        elif(estimator_type == 'pytorch'):
            from azureml.train.dnn import PyTorch
            estimator = PyTorch(**constructor_parameters)

        # Submit training
        run = self.__experiment.submit(estimator)
        print(run.get_portal_url())

        if(show_widget):
            RunDetails(run).show()

    # protected implementation methods
    def _log_metrics(self, metric_name: str, metric_value: float, description:str = None):
        print(metric_name, metric_value) 

        self.__current_run.log(metric_name, metric_value, description=description)

    
    def _complete_run(self):
        '''
        Completes the current run
        '''
        self.__current_run.complete()

    def _log_confmatrix(self, confusion_matrix: np.array, class_names: np.array):
        data = {}
        data['schema_type'] = 'confusion_matrix'
        data['schema_version'] = 'v1'
        data['data'] = {}
        data['data']['class_labels'] = class_names.tolist()
        data['data']['matrix'] = confusion_matrix.tolist()
        
        print(confusion_matrix)

        json_data = json.dumps(data)
        self.__current_run.log_confusion_matrix('Confusion matrix', json_data, description='')

    def _save_roc_curve(self, roc_auc: float, roc_plot: plt):
        self._log_metrics('roc_auc', roc_auc)
        self.__current_run.log_image('ROC Curve', plot=plt)

    def __check_compute_target(self, compute_target, use_gpu: bool):
        __vm_size = ''
        if isinstance(compute_target, AmlCompute):
            __vm_size = compute_target.vm_size
        elif isinstance(compute_target, str):
            compute = ComputeTarget(workspace=self.__workspace, name=compute_target)
            __vm_size = compute.vm_size

        if self.__vm_size_list is None:
            self.__vm_size_list = AmlCompute.supported_vmsizes(self.__workspace)
        
        vm_description = list(filter(lambda vmsize: str.upper(vmsize['name']) == str.upper(__vm_size), self.__vm_size_list))[0]
        if(use_gpu and vm_description['gpus'] == 0):
            raise errors.TrainingComputeException(f'gpu_compute was specified, but the target does not have GPUs: {vm_description} ')
        if(not (use_gpu) and vm_description['vCPUs'] == 0):
            raise errors.TrainingComputeException(f'cpu_compute was specified, but the target does not have CPUs: {vm_description} ')


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

