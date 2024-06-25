import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import Custom_Exception

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise Custom_Exception(e,sys)
