import os

def initProjectStructure():
    '''Initialize the folder strucure for the Data/AI Project that facilitates Azure MLOps 
    Args:
        No Arguments
    Returns: 
        Void in case of success and throws an error if the directory creation fails.
    '''
    folderNames = ['config','notebook','data','script','output/model','output/script']
    for folderName in folderNames:    
        try:
            if not os.path.exists(folderName):
                os.makedirs(folderName)
        except OSError:
            print ('Error: Creating directory. ' +  folderName)