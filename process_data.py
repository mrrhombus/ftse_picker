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

api_key=open(api_key_file, 'r').readlines()[0]
data_df_src=pd.read_csv(data_filepath)

tickers=data_df_src['Stock'].unique()

#%%

analysis_run=process_engine.ProcessEngine()
analysis_run.set_stock_universe(['VOD'])
analysis_run.download_data(api_key, 'full')
analysis_run.process_data()

#%%
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

analysis_run.prepare_data_for_model(keep_cols, y_col, oh_cols)
analysis_run.train_ml_predictor()
predictions=analysis_run.make_predictions(analysis_run.x_data_oh)
#%%

#%%
data_df_processed=pd.read_csv(os.path.join(data_folder,'processed_data.csv'))


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


