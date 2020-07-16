#%%

import sys
import os
import pandas as pd
import datetime
import numpy as np
import time


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder=os.path.join(ROOT_DIR, 'data')
data_filepath=os.path.join(data_folder,'timesseriesdata.csv')

sys.path.append(ROOT_DIR)
from func_lib import process_engine

api_key_file=os.path.join(data_folder, 'api_key.txt')

api_key=open(api_key_file, 'r').readlines()[0


