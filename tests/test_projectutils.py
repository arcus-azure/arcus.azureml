import os
import pathlib
import shutil
import logging
from arcus.azureml import projectutils as pu

logger = logging.getLogger()
tmp_test_path = os.path.join(os.getcwd(), 'pytestTempFolder')
test_root_path = os.getcwd()
test_foldername = 'src'

#Internal method to create the temporary test directory
def create_temp_testpath():
   if os.path.exists(tmp_test_path):
      shutil.rmtree(tmp_test_path)
   os.mkdir(tmp_test_path)
   os.chdir(tmp_test_path)

#Internal method to remove the temporary test directory
def remove_temp_testpath():
   os.chdir(test_root_path)
   shutil.rmtree(tmp_test_path)

#Call init_project_structure(root_folder) with root_folder as 'src'
def test_initprojectstructure_with_rootfolder():          
   create_temp_testpath()
   pu.init_project_structure(test_foldername)   
   count_rootfolder = len(next(os.walk(tmp_test_path))[1])
   logger.info(count_rootfolder)
   new_sample_path = os.path.join(tmp_test_path,test_foldername)   
   count_folders = len(next(os.walk(new_sample_path))[1])
   logger.info(count_folders)
   assert(count_rootfolder==1 and count_folders==5)
   remove_temp_testpath()

#Call init_project_structure(root_folder) without root_folder
def test_initprojectstructure_empty_rootfolder():   
   create_temp_testpath() 
   pu.init_project_structure()
   count_folders = len(next(os.walk(tmp_test_path))[1])   
   logger.info(count_folders)
   assert(count_folders==5)
   remove_temp_testpath()

#Call init_project_structure(root_folder) multiple times and check folders created
def test_initprojectstructure_multipletimes():         
   create_temp_testpath()
   pu.init_project_structure(test_foldername)
   pu.init_project_structure(test_foldername)
   pu.init_project_structure(test_foldername)
   count_rootfolder = len(next(os.walk(tmp_test_path))[1])
   logger.info(count_rootfolder)
   new_sample_path = os.path.join(tmp_test_path,test_foldername)   
   count_folders = len(next(os.walk(new_sample_path))[1])
   logger.info(count_folders)
   assert(count_rootfolder==1 and count_folders==5)
   remove_temp_testpath()

#Call init_project_structure(root_folder) and check if the currentdirectory is the same
def test_initprojectstructure_currdir_check():   
   create_temp_testpath()   
   currdir_before = os.getcwd()
   #logger.info(currdir_before)   
   pu.init_project_structure()
   currdir_after = os.getcwd()
   #logger.info(currdir_after)
   assert(currdir_before,currdir_after)
   remove_temp_testpath()