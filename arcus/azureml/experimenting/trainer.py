from abc import ABCMeta, abstractmethod
import pandas as pd 
import numpy as np


class Trainer:
    __metaclass__ = ABCMeta

    def new_run(self, description: str = '') :
        raise NotImplementedError
    