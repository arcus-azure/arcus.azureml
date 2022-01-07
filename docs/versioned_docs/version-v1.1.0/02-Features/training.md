---
title: "Asynchronous trainings"
layout: default
---

# Asynchronous trainings

With `arcus-azureml`, we offer an out of the box library that allows you to build a training package and execute it on a container infrastructure.  This works locally and in the cloud (on the AzureML Compute)  

1. We provide __standard scripts__ that allow you to start writing your training, without having to tinker too much with requirements and best practices.
1. The standard scripts allow by default access to the __datasets__ in the Azure ML workspace.
1. The training results and model will automatically be persisted in the AzureML Run
1. All needed requirements will be deployed on the docker image.
1. The training can be executed on local compute first (easier to test and debug) and be scheduled on the AmlCompute just by switching 1 parameter.

## Common imports

The following imports are advised for most cases in interactive experimentation mode.

```python
from arcus.azureml.environment.aml_environment import AzureMLEnvironment
```

## Step by step guide

### Create a new training package

As we will be launching our training on a docker environment, this environment has to be prepared.  Therefore, the following code will automatically create a new directory (using the experiment name) that contains two files:

- `train.py`: this file is a standard training file that contains a lot of code so that you can just focus on accessing your training data, building your model and you can get started.
- `requirements.txt`: this file contains the default packages that are required to run the training on the docker compute.  When you are using other packages, these can easily be added in this file, which will result in the building of a new docker image, during the training.  (so expect a longer training time, when this file changes)

```python
work_env = AzureMLEnvironment.Create()
training_name = 'my-training'
trainer = work_env.start_experiment(training_name)
trainer.setup_training(training_name, overwrite=False)
```

### Edit your training script

Navigate to the newly created folder and edit the `train.py` script.  This script contains the following sections that provide standard functionality.

__Argument parsing__

It is possible to submit arguments to a training script.  These arguments will be easily accessible in the training script, by using the code below.  By default it is already possible to take the `epochs`, `batch_size`, `es_patience` and `train_test_split_ratio` arguments that can be used to influence the specific training of the model.  

```python
parser = argparse.ArgumentParser()

# If you want to parse arguments that get passed through the estimator, this can be done here
parser.add_argument('--epochs', type=int, dest='epochs', default=10, help='Epoch count')
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32, help='Batch size')
parser.add_argument('--es_patience', type=int, dest='es_patience', default=-1, help='Early stopping patience. If less than zero, no Early stopping')
parser.add_argument('--train_test_split_ratio', type=float, dest='train_test_split_ratio', default=0.3, help='Train test split ratio')

args, unknown = parser.parse_known_args()
epoch_count = args.epochs
batch_size = args.batch_size
es_patience = args.es_patience
train_test_split_ratio = args.train_test_split_ratio
```
__Connectivity objects__

The following section provides a developer access to the WorkEnvironment (where datasets can be loaded, for example) as well as the trainer (where evaluation can happen):

```python
# Load the environment from the Run context, so you can access any dataset
aml_environment = AzureMLEnvironment.CreateFromContext()
trainer = AzureMLTrainer.CreateFromContext()
```

__Data access__

- __File data sets__ will be available in a directory relative to the training folder, with a name that equals the name of the dataset.  (hyphens in the dataset are however not supported).  
- __Tabular data sets__ can be access through the `AzureMLEnvironment` class, as described in [Interactive experimentations](experimenting)

```python
df = work_env.load_tabular_dataset('datasetname')
```

__Model evaluation__

It's important to track the evaluation results of the training.  This can be done through the AzureMLTrainer class as described in [Interactive experimentations](experimenting)

An example to evaluate a classifier.  This will upload the confusion matrix, the RoC curve and the metrics to the Run.

```python
trainer.evaluate_classifier(logreg, X_test, y_test, show_roc = True, upload_model = True)
```

__Add metrics to the run__

Not every model will be a classifier or a standard model that we provide an out of the box evaluation method for.  Sometimes, you will need to add your own metrics to the Run.  This can be done through the following code:

```python
# Tracking the dice coefficient
trainer._log_metrics('dice_coef_loss', list(fitted_model.history.history['dice_coef_loss'])[-1], description='')
```

__Model persistance__

After the model has been trained, it can easily be saved in the `outputs` folder.  This folder will automatically be persisted onto the AzureML workspace, so that the model can be retrieved later.

```python
fitted_model.save('outputs/model')
```

### Launch training

Once the training script is ready, it's time to start the training on the compute of choice.  
We are launching the training, by using the Estimators that are available in AzureML. 
The following steps will take place:

1. Based on the specified trainingtype (Tensorflow, Scikit-learn, Pytorch, etc), a base docker image will be taken from the public Microsoft container repository.  The following environment_types are currently supported: 
    - `tensorflow`: has tensor flow enabled and installed (supporting GPU and CPU) and can be used for Keras too
    - `sklearn`: has the scikit-learn packages installed that are providing most common Machine Learning algorithms
    - `pytorch`: has the Deep Learning package of PyTorch installed.
    - `None`: in this case an empty, default Estimator will be taken and all packages have to be provided through the requirements.txt file.
1. In extra docker layers, there will be other python packages installed, by leveragin the configured requirements.txt file
1. Once the image is ready, a container instance will be created on the specified compute.
1. The file data sets will be mounted or downloaded (depending on the configuration) and made available on the docker image (as a relative folder to the training script)
1. The training script will be launched with the script arguments 
1. After the script completes, all files that have been written to the logs and the outputs folders will be uploaded to the Run history in AzureML
1. The container instance will be removed, but the container image will remain in the Azure Container Registry that comes with an AzureML workspace (and is visible in your resource group).  This means that next time, the same image can be reused if the requirements.txt haven't changed.

__Launching a training on local compute__

The following code snippet launches a training on the local compute.  In this case the docker image will be built and started on the local host, where docker is required to be running.  All logs will be made visible in the Widget, visible in the notebook.  The process of that run can also be monitored in the AzureML Workspace portal.

```python
arguments = {
    '--epochs': 5,
    '--filter_count': 5,
    '--max_images': 20
}

trainer.start_training(training_name, environment_type='tensorflow', 
                       compute_target='local', gpu_compute=True, script_parameters = arguments)
```

__Launching a training on AmlCompute__

The following code snippet launches the training on a compute target (indicated by the `compute_target` argument) in the AzureML workspace.  The same mechanics are at play as described in the previous section. 

```python
arguments = {
    '--epochs': 5,
    '--filter_count': 5,
    '--max_images': 20
}

trainer.start_training(training_name, environment_type='tensorflow', 
                       compute_target='gpu-cluster', gpu_compute=True, script_parameters = arguments)
```

__Working with files in a training__

Quite often there is a need to leverage files in a training.  Thinking about neural networks that require images in their training and test data sets, for example.  The following steps describe two options to make files available on the training containers.

- A `FileDataSet` can be mounted to an Estimator.  By mounting a dataset, a reference is being made on the training cluster that references the files in the cloud.  The files are only downloaded or accessed when the script is performing an action (listing files in the folder, reading a file by name, etc).  
- A `FileDataSet`can be downloaded to an Estimator.  By downloading a dataset, all files are fully downloaded onto the training cluster (in the training folder).  The initial startup of your training might take longer, as this only happens after all files have been downloaded.  

The following code snippet shows to datasets that are being made available on the training cluster.  One is mounted, the other is being downloaded.  But the actual access to the files happens exactly the same across both methods, as you can see in the training script.  The files are made available in a directory with the exact same name as the dataset.

```python
mount_dataset_name = 'ds_to_mount'
download_dataset_name = 'ds_to_download'

trainer.start_training(training_name, environment_type='tensorflow', 
                       input_datasets = [mount_dataset_name],
                       input_datasets_to_download = [download_dataset_name],
                       compute_target='gp-cluster', gpu_compute=True, script_parameters = arguments)

```
