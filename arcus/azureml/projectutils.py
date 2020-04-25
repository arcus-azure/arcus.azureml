'''
The projectutils module provides common operations related to data/ai projects
'''

import os

def initProjectStructure():
    '''Creates the folders to start a the Data/AI Project that facilitates Azure MLOps
       Skips folder creation if folder already exists. 
    Args:
        No Arguments
    Returns: 
        Void in case of success and throws an error if the directory creation fails.
    '''
    folderNames = ['config','notebook','data','script','output/model','output/script']
    currentPath = os.getcwd()
    for folderName in folderNames:    
        try:
            if not os.path.exists(folderName):                
                os.makedirs(os.path.join(currentPath, folderName))
        except OSError:
            print ('Error: Creating directory. ' +  folderName)