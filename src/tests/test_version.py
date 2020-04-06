import sys
import os
import pytest
import arcus.azureml.version as vr

def test_base():
    assert (2+3) == 5

def test_version_out():
    assert (vr.output() == '0.0.1')