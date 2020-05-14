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

class AzureMLTrainer(trainer.Trainer):
    is_connected: bool = False
    __config_file: str = '.azureml/config.json'
    __workspace: Workspace = None
    __experiment: Experiment = None
    __current_experiment_name: str
    __current_run: Run = None

    def __init__(self, experiment_name: str, aml_workspace: Workspace):
        '''
        Initializes a new connected Trainer that will persist and log all runs on AzureML workspace
        Args:
            experiment_name (str): The name of the experiment that will be seen on AzureML
            aml_workspace (Workspace): The connected workspace on AzureML
        '''
        self.__workspace = aml_workspace
        self.__current_experiment_name = experiment_name

        self.__experiment = Experiment(workspace=self.__workspace, name=experiment_name)

    def new_run(self, description: str = None) -> Run:
        '''
        This will begin a new interactive run on the existing AzureML Experiment.  When a previous run was still active, it will be completed.
        Args:
            description (str): An optional description that will be added to the run metadata
        Returns:
            Run: the AzureML Run object that can be used for further access and custom logic
        '''
        if(self.__current_run is not None):
            self.__current_run.complete()
        self.__current_run = self.__experiment.start_logging()
        if(description is not None):
            self.__current_run.log('Description', description)
        
        return self.__current_run

    def evaluate_classifier(self, fitted_model, X_test: np.array, y_test: np.array, show_roc: bool = False, 
                            class_names: np.array = None, finish_existing_run: bool = True, upload_model: bool = True) -> np.array:
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
        Returns: 
            np.array: The predicted (y_pred) values against the model
        '''
        y_pred = fitted_model.predict(X_test)
        if class_names is None:
            class_names = np.char.mod('%d', sorted(np.unique(y_test)))

        print(metrics.classification_report(y_test, y_pred))

        cf = metrics.confusion_matrix(y_test, y_pred)
        self.__log_confmatrix(cf, class_names)
        print(cf)
        accuracy = metrics.accuracy_score(y_test, y_pred) * 100
        self.__current_run.log('accuracy', accuracy, description='')
        print('Accuracy score:', accuracy) 

        if(show_roc == True):
            # Verify that we are having a binary classifier
            if(len(fitted_model.classes_)!=2):
                raise AttributeError('Showing a ROC curve is only possible for binary classifier, not for multi class')
            self.__log_roc_curve(y_test, y_pred) 

        if upload_model:
            # Save the model to the outputs directory for capture
            model_file_name = 'outputs/model.pkl'
            joblib.dump(value = fitted_model, filename = model_file_name)
            self.__current_run.upload_file(name = model_file_name, path_or_stream = model_file_name)

        if (finish_existing_run and self.__current_run is not None):
            self.__current_run.complete(True)

        return y_pred

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

    def __log_confmatrix(self, confusion_matrix: np.array, class_names: np.array):
        data = {}
        data['schema_type'] = 'confusion_matrix'
        data['schema_version'] = 'v1'
        data['data'] = {}
        data['data']['class_labels'] = class_names.tolist()
        data['data']['matrix'] = confusion_matrix.tolist()

        json_data = json.dumps(data)
        self.__current_run.log_confusion_matrix('Confusion matrix', json_data, description='')

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
        #plt.show()
        #figure_name = self.__current_run.id + '_roc.png'
        #plt.savefig(figure_name)
        self.__current_run.log('roc_uac', roc_auc)
        self.__current_run.log_image('ROC Curve', plot=plt)
        plt.close()
        return roc_auc