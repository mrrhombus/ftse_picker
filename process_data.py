#%%

import sys
import os
import pandas as pd
import datetime
import numpy as np
import math
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#%%
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder=os.path.join(ROOT_DIR, 'data')
data_filepath=os.path.join(data_folder,'timesseriesdata.csv')

data_df_src=pd.read_csv(data_filepath)
#%%
#trim data during dev




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

def create_date_shift_cols(data_df):
    temp_df=data_df.copy()
    temp_df.sort_values('date')
    temp_df['date_tm1']=temp_df['date'].shift(-1)
    temp_df['date_tp1']=temp_df['date'].shift(1)
    return temp_df

def create_forward_and_back_looking_returns(data_df):
    out_df=data_df.copy()
    temp_df=data_df.loc[:,['date','close']]
    temp_df=temp_df.rename(columns={'close':'close_tp1','date':'date_tp1'})
    out_df=pd.merge(out_df, temp_df, how='inner',on='date_tp1')
    temp_df=data_df.loc[:,['date','close']]
    temp_df=temp_df.rename(columns={'close':'close_tm1','date':'date_tm1'})
    out_df=pd.merge(out_df, temp_df, how='inner',on='date_tm1')
    out_df['return_nextDay']=out_df.apply(calc_return, axis=1, args=['close_tp1', 'close'])
    out_df['return_backwards']=out_df.apply(calc_return, axis=1, args=['close', 'close_tm1'])
    return out_df

def add_extra_stats(data_df):
    data_df=data_df.copy()
    data_df['day_of_week']=data_df['date'].apply(lambda x: x.weekday())
    data_df['month']=data_df['date'].apply(lambda x: x.month)
    data_df['year']=data_df['date'].apply(lambda x: x.year)
    data_df['day_of_month']=data_df['date'].apply(lambda x: x.day)
    
    data_df['high_low_gap']=data_df['high']-data_df['low']
    
    data_df['high_low_gap_percentile']=data_df.apply(stat_percentile_finder, axis=1, args=['high_low_gap',data_df])
    data_df['volume_percentile']=data_df.apply(stat_percentile_finder, axis=1, args=['volume',data_df])
    return data_df

def lag_col(data_df, order_by, lag_col, shift_size, new_col_name):
    temp_df=data_df.copy()
    temp_df.sort_values(order_by)
    temp_df[new_col_name]=temp_df[lag_col].shift(shift_size)    
    return temp_df

def add_lag_returns(data_df):
    data_df=data_df.copy()
    for t in np.arange(1,21,1):
        base_col='return_backwards'        
        new_col='return_backwards_tm{0}'.format(t)
        data_df=lag_col(data_df, 'date',base_col, -1*t,new_col)        
    return data_df

def replace_catcols_oh(x_df, oh_cols):
    my_oh=OneHotEncoder(handle_unknown='ignore')

    for c in oh_cols:   
        data_mat=np.matrix(x_df[c])
        my_oh.fit(data_mat.transpose())
        e=my_oh.transform(data_mat.transpose())
        e=e.todense()
        new_col_names=my_oh.get_feature_names([c])
        x_df=x_df.drop(columns=(c))

        for cn in range(new_col_names.__len__()):
            new_col_name=new_col_names[cn]
            x_df[new_col_name]=e[:,cn]
    return x_df
#%%
    
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


tickers=data_df_src['Stock'].unique()
ticker=tickers[0]

data_df_processed=pd.DataFrame()

for ticker in tickers:
    try:
        t1=time.time()
        data_df=data_df_src.loc[data_df_src['Stock']==ticker,:]
        batch_size=data_df.shape[0]
        data_df=data_df.copy()
        data_df['date']=data_df['timestamp'].apply(parse_date)
        data_df=create_date_shift_cols(data_df)
        data_df=create_forward_and_back_looking_returns(data_df)
        data_df=add_extra_stats(data_df)
        data_df=add_lag_returns(data_df)
        
        t2=time.time()
        print('{2} {1} rows complete in {0}'.format((t2-t1),batch_size, ticker))
        
        data_df_processed=pd.concat([data_df_processed, data_df],axis=0)
    except:
        print('failed to process {0}'.format(ticker))

data_df_processed.to_csv(os.path.join(data_folder,'processed_data.csv'))




#%%
data_df_processed=pd.read_csv(os.path.join(data_folder,'processed_data.csv'))



#%%
keep_cols=['Stock','return_nextDay', 
           'return_backwards', 'day_of_week', 'month', 'year', 'day_of_month',
           'high_low_gap', 'high_low_gap_percentile', 'volume_percentile',
           'return_backwards_tm1',  'return_backwards_tm2', 'return_backwards_tm3', 
           'return_backwards_tm4', 'return_backwards_tm5', 'return_backwards_tm6', 
           'return_backwards_tm7',  'return_backwards_tm8', 'return_backwards_tm9',  
           'return_backwards_tm10', 'return_backwards_tm11', 'return_backwards_tm12',  
           'return_backwards_tm13', 'return_backwards_tm14', 'return_backwards_tm15',  
           'return_backwards_tm16', 'return_backwards_tm17', 'return_backwards_tm18',  
           'return_backwards_tm19','return_backwards_tm20']    
data_df=data_df_processed.loc[:,keep_cols]

#%%
for c in keep_cols:
    w=pd.isna(data_df[c])
    data_df=data_df.loc[-w,:]
    if sum(w)>0:
        print('col {0} has {1} na entries'.format(c, sum(w)))

#%%
x_cols=[c for c in data_df.columns if c !='return_nextDay']
y_cols='return_nextDay'
x_data=data_df.loc[:,x_cols]
y_data=data_df[y_cols]

#%%
#one hot stock


oh_cols=['Stock',
         'day_of_week', 
         'month', 
         'year', 
         'day_of_month',]

x_data_oh=replace_catcols_oh(x_data, oh_cols)

#%%


x_train, x_valid, train_y, val_y = train_test_split(x_data_oh, y_data, random_state=1)

#%%
# Using best value for max_leaf_nodes
#test_cols=['day_of_week']
x_train_test=x_train.copy()
x_valid_test=x_valid.copy()
        
ftsepicker_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
ftsepicker_model.fit(x_train_test, train_y)
test_predictions=ftsepicker_model.predict(x_train_test)
train_mae = mean_absolute_error(test_predictions, train_y)
print("Train MAE for best value of max_leaf_nodes: {:,.5f}".format(100*train_mae))

val_predictions = ftsepicker_model.predict(x_valid_test)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.5f}".format(100*val_mae))
#%%
from matplotlib import pyplot as plt

plt.scatter(val_predictions, val_y)
plt.show()
#%%
predict_all=ftsepicker_model.predict(x_data_oh)
results_analysis=x_data.copy()
results_analysis['y']=y_data
results_analysis['y_prediction']=predict_all
results_analysis['abs_error']=abs(results_analysis['y']-results_analysis['y_prediction'])
#%%
summary_error=pd.pivot_table(results_analysis,
                        values='abs_error',
                        index='Stock',
                        aggfunc=np.mean)
summary_vol=pd.pivot_table(results_analysis,
                        values='y',
                        index='Stock',
                        aggfunc=np.std)
summary=pd.merge(summary_error, summary_vol, left_index=True, 
right_index=True)

summary['rel_error']=summary['abs_error']/summary['y']
summary['Stock']=summary.index
summary=summary.sort_values('rel_error')
summary.index=np.arange(summary.shape[0])
best_fit_stocks=summary.loc[np.arange(50),'Stock']

w=results_analysis['Stock'].isin(best_fit_stocks)
plt.scatter(results_analysis.loc[w,'y'],
            results_analysis.loc[w,'y_prediction'])

#%%
plt.hist(summary['abs_error'])


