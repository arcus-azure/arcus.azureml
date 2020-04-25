import os
import shutil
from arcus.azureml.projectutils import initProjectStructure
tmpTestPath = os.path.join(os.getcwd(), 'pytestTempFolder')

def createTempTestPath():   
   if os.path.exists(tmpTestPath):
      shutil.rmtree(tmpTestPath)
   os.mkdir(tmpTestPath)

def test_initProjectStructure():
   testRootPath = os.getcwd()
   createTempTestPath()   
   os.chdir(tmpTestPath)   
   initProjectStructure()
   os.chdir(testRootPath)   
   assert(len(next(os.walk(tmpTestPath))[1]) == 5)
   shutil.rmtree(tmpTestPath)   