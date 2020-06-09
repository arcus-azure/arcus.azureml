---
title: "Asynchronous trainings"
layout: default
---

# Asynchronous trainings

With `arcus-azureml`, we offer an out of the box library that allows you to build a training package and execute it on a container infrastructure.  This works locally and in the cloud (on the AzureML Compute)  

1. We provide __standard scripts__ that allow you to start writing your training, without having to tinker too much with requirements and best practices.
1. The standard scripts allow by default access to the __datasets__ in the Azure ML workspace.
1. The training results and model will automatically be persisted in the AzureML Run
1. All required requirements will be deployed on the docker image.
1. The training can be executed on local compute first (easier to test and debug) and be scheduled on the AmlCompute just by switching 1 parameter.

## Common imports

The following imports are advised for most cases in interactive experimentation mode.

```python
from arcus.azureml.environment.environment_factory import WorkEnvironmentFactory as factory
```

## Step by step guide


## Sample notebooks



[&larr; back](/)