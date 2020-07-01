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

def find_prior_date(x, data_df, date_col, filter_rows):
    temp_df=data_df.copy()
    for f in filter_rows.keys():
        w=temp_df[f]==filter_rows[f]
        temp_df=temp_df.loc[w,:]
    date_list=temp_df[date_col]
    w=date_list<x    
    if sum(w)<1:
        return None
    last_date=max(date_list[w])
    return last_date
def prior_date_finder(row, data_df):
    ticker=row['Stock']
    date=row['date']
    prior_date=find_prior_date(date, data_df, 'date', {'Stock':ticker})
    return prior_date

data_df['date']=data_df['timestamp'].apply(parse_date)
data_df['date_tm1']=data_df.apply(prior_date_finder, axis=1, args=[data_df])
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





# %%
