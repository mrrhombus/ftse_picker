#%%

import sys
import os
import pandas as pd
import datetime
import numpy as np
import math
import time


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder=os.path.join(ROOT_DIR, 'data')
data_filepath=os.path.join(data_folder,'timesseriesdata.csv')

sys.path.append(ROOT_DIR)
from func_lib import ftsepicker_lib

#%%

data_df_src=pd.read_csv(data_filepath)


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
        data_df=ftsepicker_lib.create_date_shift_cols(data_df)
        data_df=ftsepicker_lib.create_forward_and_back_looking_returns(data_df)
        data_df=ftsepicker_lib.add_extra_stats(data_df)
        data_df=ftsepicker_lib.add_lag_returns(data_df)
        
        t2=time.time()
        print('{2} {1} rows complete in {0}'.format((t2-t1),batch_size, ticker))
        
        data_df_processed=pd.concat([data_df_processed, data_df],axis=0)
    except:
        print('failed to process {0}'.format(ticker))

data_df_processed.to_csv(os.path.join(data_folder,'processed_data.csv'))




#%%
data_df_processed=pd.read_csv(os.path.join(data_folder,'processed_data.csv'))
y_col='return_nextDay'

oh_cols=['Stock',
            'day_of_week', 
            'month', 
            'year', 
            'day_of_month',]

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


x_data, x_data_oh, y_data = ftsepicker_lib.prepare_dataset_dectree(data_df_processed,
                                                            keep_cols,
                                                            y_col,
                                                            oh_cols)


#%%
for leaf_nodes in [10,20,50,80,100,150,200]:
    print('Using {0} leaf nodes'.format(leaf_nodes))
    ftse_stock_picker_model=ftsepicker_lib.train_and_predict(x_data_oh,
                                                         y_data,
                                                        leaf_nodes)



#%%
# Using best value for max_leaf_nodes
#test_cols=['day_of_week']


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


