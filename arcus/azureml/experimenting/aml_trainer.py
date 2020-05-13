from arcus.azureml.experimenting import trainer
from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
from azureml.data.datapath import DataPath
from azureml.data.dataset_error_handling import DatasetValidationError, DatasetExecutionError
from azureml.data.dataset_type_definitions import PromoteHeadersBehavior
from datetime import datetime
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import json

class AzureMLTrainer(trainer.Trainer):
    is_connected: bool = False
    __config_file: str = '.azureml/config.json'
    __workspace: Workspace = None
    __experiment: Experiment = None
    __current_experiment_name: str
    __current_run: Run = None

    def __init__(self, experiment_name: str, aml_workspace: Workspace):
        self.__workspace = aml_workspace
        self.__current_experiment_name = experiment_name

        self.__experiment = Experiment(workspace=self.__workspace, name=experiment_name)

    def new_run(self, description: str = None) -> Run:
        if(self.__current_run is not None):
            self.__current_run.complete()
        self.__current_run = self.__experiment.start_logging()
        if(description is not None):
            self.__current_run.log('Description', description)
        
        return self.__current_run

    def evaluate_classifier(self, fitted_model, X_test: np.array, y_test: np.array, show_roc: bool = False, class_names: np.array = None, finish_existing_run: bool = True) -> np.array:
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
        
        if (finish_existing_run and self.__current_run is not None):
            self.__current_run.complete(True)

        return y_pred

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
        '''Will plot the Receiver Operating Characteristic (ROC) Curve for binary classifiers

        Args:
            y_pred (np.array): The predicted values of the test set 
            y_test (np.array): The actual outputs of the test set

        Returns: 
            float: The ROC_AUC value
        '''
        # calculate the fpr and tpr for all thresholds of the classification
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
        roc_auc = metrics.auc(fpr, tpr)

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

        return roc_auc