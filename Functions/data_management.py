import os 

# Methode to find all files in subfolders of directory.
# This methode is used to find all rawdata in subfolders of folder handed in. 
# Be cautions that folder you handed over is the right one. 
def find_all_files_in_dir(dir_data: str):
    '''
    Methode to find all files in folder

    Parameters
    ----------
    dirertory : string
        Main folder in which raw data is stored in. 
        
    Returns
    -------
    file_list : list
        list of the directories of all files in folder itself and of subfolders. 

    '''
    
    list_of_subpaths = os.listdir(dir_data)
    files = list()
    
    # Search iteravely in all subfolders
    for sub_path in list_of_subpaths:
        
        current_path = os.path.join(dir_data, sub_path)
        
        # iterate procedure, if directory is folder. Otherwise store in files  
        if os.path.isdir(current_path):
            files = files + find_all_files_in_dir(current_path)
        else:
            files.append(current_path)
                
    return files