# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:26:34 2020

@author: pr967
"""
import sys
import time
import requests
import pandas as pd
import csv
import io
import os
api_key=r'3O7TSALOH8TXAHNM'
project_dir=r'C:\Users\pr967\Documents\FTSE350 picker'
tickers=pd.read_csv(os.path.join(project_dir,'ftse350tickers.csv'))


url=r'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey={0}&datatype=csv'.format(api_key)
r=requests.get(url)
data=pd.read_csv(io.StringIO(r.content.decode('utf-8')))

def urlBuilder(ticker):
    url=r'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={1}&outputsize=full&apikey={0}&datatype=csv'.format(api_key,ticker)
    return url

summary_df=pd.DataFrame()
fail_list=[]
for t in tickers['Tickers']:
    url=urlBuilder(t)
    r=requests.get(url)
    time.sleep(12)
    df=pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    if df.shape[1]==6:
        df['Stock']=t
        summary_df=pd.concat([summary_df, df], axis=0)
    else:
        print('failed to download {0}'.format(t))
        fail_list.append(t)
    

summary_df.to_csv(os.path.join(project_dir, 'timesseriesdata.csv'), index=False)

#%%
fail_tickers=pd.DataFrame({'Tickers':fail_list})
fail_tickers.to_csv(os.path.join(project_dir, 'missingtickers.csv'), index=False)
