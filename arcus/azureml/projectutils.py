'''
The projectutils module provides common operations related to data/ai projects
'''
import os
from azureml.core import Workspace

def init_project_structure(root_folder:str = None):
    '''Creates the folders to start a the Data/AI Project that leverages Azure MLOps
       Skips folder creation if folder already exists. 
    Args:
        root_folder (str): The root folder where the arcus project folders has to be initialized    
    '''    
    folder_names = ['config','notebook','data','script','output/model','output/script']
    user_workdir = os.getcwd()    
    if root_folder == None:
        root_path = user_workdir
    else:
        root_path = os.path.join(os.getcwd(), root_folder)
    #Switch to root_folder, create if not exists
    if not os.path.exists(root_path):
        try:
            os.mkdir(root_path)
            os.chdir(root_path)
        except:
            print ('Error creating root directory. ' + root_path)
    else:
        print('Folder already exists :' + root_path)
    
    for folder_name in folder_names:
        try:
            folder_path = os.path.join(root_path, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        except:
            print('Error creating project directory. ' + folder_path)
        finally:
            os.chdir(user_workdir)
