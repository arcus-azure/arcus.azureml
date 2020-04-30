'''
The projectutils module provides common operations related to data/ai projects
'''
import os

def init_project_structure(root_folder:str = None):
    '''Creates the folders to start a the Data/AI Project that facilitates Azure MLOps
       Skips folder creation if folder already exists. 
    Args:
        root_folder (str): The root folder where the arcus project folders has to be initialized
    Returns: 
        Void in case of success and throws an error if the directory creation fails.
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