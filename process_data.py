#%%

import sys
import os
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

data_filepath=os.path.join(ROOT_DIR,'timesseriesdata.csv')

data_df=pd.read_csv(data_filepath)

#%%
def parse_date(d_str, splitter='-'):
    #works for yyyy-mm-dd
    d_parts=d_str.split(splitter)
    date_out=pd.datetime(int(d_parts[0]),
                         int(d_parts[1]),
                         int(d_parts[2]))
    return date_out

def find_prior_date(x, date_list, filter_rows):
    w=date_list<x
    w=w&filter_rows
    last_date=max(date_list[w])
    return last_date

data_df['date']=data_df['timestamp'].apply(parse_date)
data_df['date_tm1']=
#compute returns
#volatility

#prepare dataset
#day of week
#week of year
#year
#lag returns (20 days?)
#get ftse350 return

#external data?
#inflation data



