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
trainer.setup_training(training_name)
```

### Create your training script

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

__Model evaluation__

It's important to track the evaluation results of the training.  This can be done through the AzureMLTrainer class as described in [Interactive experimentations](experimenting)

```python
trainer.evaluate_classifier(logreg, X_test, y_test, show_roc = True, upload_model = True)
```

__Model persistance__

After the model has been trained, it can easily be saved in the `outputs` folder.  This folder will automatically be persisted onto the AzureML workspace, so that the model can be retrieved later.

```python
fitted_model.save('outputs/model')
```

## Sample notebooks



[&larr; back](/)