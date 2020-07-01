#%%

import sys
import os
import pandas as pd
import datetime
import numpy as np
import math

#%%
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

data_filepath=os.path.join(ROOT_DIR,'timesseriesdata.csv')

data_df_src=pd.read_csv(data_filepath)
#%%
#trim data during dev

data_df=data_df_src.loc[np.arange(1000),:]


def parse_date(d_str, splitter='-'):
    #works for yyyy-mm-dd
    d_parts=d_str.split(splitter)
    date_out=datetime.date(int(d_parts[0]),
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

def prior_date_finder(row, date_col, data_df):
    ticker=row['Stock']
    date=row[date_col]
    prior_date=find_prior_date(date, data_df, 'date', {'Stock':ticker})
    return prior_date

def get_percentile(x,x_list):
    w=x_list<=x
    p=float(sum(w)+1)/float(x_list.__len__()+1)
    return p
    
def stat_percentile_finder(row, stat_col, data_df):
    ticker=row['Stock']
    measure=row[stat_col]
    w=data_df['Stock']==ticker
    stats=data_df.loc[w,stat_col]
    p=get_percentile(measure, stats)
    return p

def calc_return(row, p_t, p_tm1):
    log_ret=math.log(row[p_t]/row[p_tm1])
    return log_ret

data_df['date']=data_df['timestamp'].apply(parse_date)
data_df['date_tm1']=data_df.apply(prior_date_finder, axis=1, args=['date', data_df])
temp_df=data_df.loc[:,['date','date_tm1','close']]
temp_df=temp_df.rename(columns={'close':'close_tp1','date_tm1':'date', 'date':'date_tp1'})
data_df=pd.merge(data_df, temp_df, how='inner',on='date')
data_df['return_nextDay']=data_df.apply(calc_return, axis=1, args=['close_tp1', 'close'])

temp_df=data_df.loc[:,['date','date_tm1','close']]
temp_df=temp_df.rename(columns={'close':'close_tp1','date_tm1':'date', 'date':'date_tp1'})
data_df=pd.merge(data_df, temp_df, how='inner',on='date')
data_df['return_nextDay']=data_df.apply(calc_return, axis=1, args=['close_tp1', 'close'])

data_df['day_of_week']=data_df['date'].apply(lambda x: x.weekday())
data_df['month']=data_df['date'].apply(lambda x: x.month)
data_df['year']=data_df['date'].apply(lambda x: x.year)
data_df['day_of_month']=data_df['date'].apply(lambda x: x.day)

data_df['high_low_gap']=data_df['high']-data_df['low']

data_df['high_low_gap_percentile']=data_df.apply(stat_percentile_finder, axis=1, args=['high_low_gap',data_df])
data_df['volume_percentile']=data_df.apply(stat_percentile_finder, axis=1, args=['volume',data_df])

#%%
for t in np.arange(1,21,1):
    base_date='date_tm{0}'.format(t-1)
    if t==1:
        base_date='date'
    new_date='date_tm{0}'.format(t)
    data_df[new_date]=data_df.apply(prior_date_finder, axis=1, args=[base_date, data_df])
    temp_df=data_df.loc[:,['date','close']]
    temp_df=temp_df.rename(columns={'close':'close_tm{0}'.format(t)})
    data_df=pd.merge(data_df, temp_df, how='inner',on='date')
    
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
