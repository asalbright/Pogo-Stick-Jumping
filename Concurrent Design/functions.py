###############################################################################
# functions.py
#
# Contains useful functions for the pogo-stick  
#
# Created: 02/23/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
# Modified:
# * 10/02/2021 - Updated saving data so it is all saved in one file
###############################################################################
# # ********************************************
# Author: Andrew Albright
# Date: 03/31/2021

# File containing useful functions

# ********************************************

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import os
import sys
import tkinter
from tkinter import filedialog

def queueDirectoryPath(current_path, header=None):
    '''
    This function queues a directory path for the user to select.

    Parameters
    ----------
    current_path : Path
        The current path to the directory
    header : str
        The header of the gui

    Returns
    -------
    Path
        The path to the directory
    '''
    if header is None:
        header = "Select the Directory Where Data is Stored"

    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    currdir = current_path
    tempdir = filedialog.askdirectory(parent=root, initialdir=currdir, title=header)
    if not os.path.exists(currdir):
        raise ValueError("\nPath does not exits!")
    elif not os.path.isdir(tempdir):
        raise ValueError("\nNot a directory!")
    elif tempdir is None:
        raise ValueError("\nNo directory selected!")
    if len(tempdir) > 0:
        print("You chose %s" % tempdir)

    return Path(tempdir)

def getFiles(file_type, path=None):
    '''
    This function gets all the files of a certain type in a directory.

    Parameters
    ----------
    file_type : str
        The file type to look for
    path : Path
        The path to the directory

    Returns
    -------
    list
        The list of file paths to the files
    path: Path
        The path to the directory if only one file is found
    '''

    if path is None:
        raise ValueError(f'Path not specified for {file_type} file finder.')
    else:
        files = glob.glob(str(path / f'*.{file_type}'))

    if len(files) == 0:
        raise ValueError(f'No {file_type} files found in {path}.')
    else:
        return files 

def getFileNames(file_type, path=None)->list:
    '''
    This function gets all the file names of a certain type in a directory.

    Parameters
    ----------  
    file_type : str
        The file type to look for
    path : Path
        The path to the directory

    Returns
    -------
    list
        The list of file names to the files
    str
        The name of the file if only one file is found
    '''

    if path is None:
        raise ValueError(f'Path not specified for {file_type} name finder.')
    else:
        files = glob.glob(str(path / f'*.{file_type}'))
        print(f'Number of {file_type} files found: {len(files)}')
        file_names = []
        for f in files:
            file_names.append(os.path.basename(f).split(sep='.')[0])
        
        if len(file_names) == 1:
            return file_names[0]
        elif len(file_names) == 0:
            raise ValueError(f'No {file_type} files found in {path}.')
        else:
            return file_names

def readCSVFiles(files)->list:
    '''
    This function reads in a list of csv files and returns a list of dataframes.

    Parameters
    ----------
    files : list
        The list of csv files to read

    Returns
    -------
    list
        The list of dataframes
    '''

    data = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.replace(' ', '')
        df.columns = df.columns.str.replace('#', '')
        data.append(df)
        
    return data

def combineDataInFolder(file_type, path=None) -> pd.DataFrame:
    '''
    This function combines all the data in a folder into one dataframe.

    Parameters
    ----------
    file_type : str
        The file type to look for
    path : Path
        The path to the directory, if None, will ask the user

    Returns
    -------
    pd.DataFrame
        The dataframe containing all the data
    '''

    if path is None:
        path = queueDirectoryPath(Path.cwd(), "Select path where data exists.")

    files = getFiles(file_type, path)
    data = None
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.replace(' ', '')
        df.columns = df.columns.str.replace('#', '')
        if not data is None:
            data = pd.concat([data, df], axis=1)
        else:
            data = df
    
    return data

def parseDataFrame(df, unique_headers):
    '''
    This function parses a dataframe into a dictionary of dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to parse
    unique_headers : list
        The list of unique headers to parse the dataframe by

    Returns
    -------
    dict
        The dictionary of dataframes
    '''

    # Make sure all columns are strings without following characters
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.replace('#', '')

    data_frames = {}
    for header in unique_headers:
        temp_df = None
        for col in df.columns:
            if header in col:
                if not temp_df is None:
                    temp_df = pd.concat([temp_df, df[col]], axis=1)
                else:
                    temp_df = df[col]
        # append to dict of dataframes
        data_frames[header] = temp_df
    
    return data_frames

def dfAverageStd(df):
    '''
    This function calculates the average and standard deviation of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the average and standard deviation of

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        The dataframes containing the average and standard deviation
    '''
    
    return df.mean(axis=1), df.std(axis=1)

# Create gui to say yes/no
def guiYesNo(header, question):
    '''
    Creates a gui to ask the user if yes or no

    Parameters
    ----------
    header : str
        The header of the gui
    question : str
        The question to ask the user

    Returns
    -------
    str
        "yes" if yes, "no" if no
    '''
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    answer = tkinter.messagebox.askquestion(f'{header}', f'{question}')
    return answer

if __name__ == "__main__":
    pass