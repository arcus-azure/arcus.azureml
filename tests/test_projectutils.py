import os
import pathlib
import shutil
import logging
from arcus.azureml.projectutils import init_project_structure
tmpTestPath = os.path.join(os.getcwd(), 'pytestTempFolder')

def create_temp_testpath():
   if os.path.exists(tmpTestPath):
      shutil.rmtree(tmpTestPath)
   os.mkdir(tmpTestPath)
   os.chdir(tmpTestPath)

def test_init_project_structure():
   logger = logging.getLogger()
   test_folderName = "sample"   
   testRootPath = os.getcwd()
   create_temp_testpath()
   init_project_structure(test_folderName)
   os.chdir(tmpTestPath)
   countRootFolder = len(next(os.walk(tmpTestPath))[1])
   logger.info(countRootFolder)
   newSamplePath = os.path.join(tmpTestPath,test_folderName)
   os.chdir(newSamplePath)
   countFolders = len(next(os.walk(newSamplePath))[1])
   logger.info(countFolders)
   assert(countRootFolder==1 and countFolders==5)
   os.chdir(testRootPath)
   shutil.rmtree(tmpTestPath)