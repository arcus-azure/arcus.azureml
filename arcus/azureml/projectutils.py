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
    folderNames = ['config','notebook','data','script','output/model','output/script']    
    rootPath = os.path.join(os.getcwd(), root_folder)    
    #Switch to RootFolder, create if not exists
    if not os.path.exists(rootPath):        
        try:
            os.mkdir(rootPath)  
            os.chdir(rootPath)
        except OSError:
            print ('Error creating root directory. ' + rootPath)
    
    for folderName in folderNames:    
        try:
            if not os.path.exists(folderName):                
                os.makedirs(os.path.join(rootPath, folderName))
        except OSError:
            print ('Error creating project directory. ' + folderName)


def connect_azuremlworkspace(subscription_id:str, resource_group:str, workspace_name:str, create_resources:bool = False):
    '''Connects to AzureML workspace and saves the config to ./config folder
       Note: This method expects that init_project_structure has already been executed
    Args:
        subscription_id (str): The root folder where the arcus project folders has to be initialized
        resource_group (str): The root folder where the arcus project folders has to be initialized
        workspace_name (str): Name of the AzureML workspace
        create_resources(bool): If new resources are to be created when does not exist
    Returns: 
        A workspace session object containing the connection
    '''    
    # Things to figure out
    # How to maintain the rootfolder in the session
    # Check on local vs azureml connection 
    ws = Workspace(subscription_id, resource_group, workspace_name)
    ws.write_config(path= configFolder, file_name="ws_config.json")


    print(subscription_id, resource_group, workspace_name)