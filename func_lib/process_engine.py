
import pandas as pd
import requests
from . import ftsepicker_lib

class ProcessEngine:
    def __init__(self):
        return

    def set_stock_universe(self, stock_list):
        self.stock_universe=stock_list
        return
    
    
    def load_timeseries_data(self, file_path):

        self.timesseries_data=pd.read_csv(data_filepath)

        return
    
    def urlBuilder(self,ticker,  size='full'):
        url=r'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={1}&outputsize={2}&apikey={0}&datatype=csv'.format(api_key,ticker, size)
        return url

    def download_data(self,api_key, download_size='compact'):
        api_key=r'3O7TSALOH8TXAHNM'

        summary_df=pd.DataFrame()
        fail_list=[]
        for t in self.stock_universe:
            try:
                url=urlBuilder(t, api_key, download_size)
                r=requests.get(url)
                time.sleep(12)
                df=pd.read_csv(io.StringIO(r.content.decode('utf-8')))
                if df.shape[1]==6:
                    df['Stock']=t
                    summary_df=pd.concat([summary_df, df], axis=0)
                else:
                    print('failed to download {0}'.format(t))
                    fail_list.append(t)
            except:
                print('failed to download {0}'.format(t))
                fail_list.append(t)
            
            self.timesseries_data=summary_df
        return

    def process_data(self):
        data_df_processed=pd.DataFrame()
        ts_data=self.timesseries_data
        tickers=ts_data['Stock'].unique()

        for ticker in tickers:
            try:
                t1=time.time()
                data_df=ts_data[ts_data['Stock']==ticker,:]
                batch_size=data_df.shape[0]
                data_df=data_df.copy()
                data_df['date']=data_df['timestamp'].apply(ftsepicker_lib.parse_date)
                data_df=ftsepicker_lib.create_date_shift_cols(data_df)
                data_df=ftsepicker_lib.create_forward_and_back_looking_returns(data_df)
                data_df=ftsepicker_lib.add_extra_stats(data_df)
                data_df=ftsepicker_lib.add_lag_returns(data_df)
                
                t2=time.time()
                print('{2} {1} rows complete in {0}'.format((t2-t1),batch_size, ticker))
                
                data_df_processed=pd.concat([data_df_processed, data_df],axis=0)
            except:
                print('failed to process {0}'.format(ticker))
            
            self.processed_dataset=data_df_processed

            
        return
    
    def prepare_data_for_model(self, keep_cols, y_col, oh_cols):
        
        self.x_data, self.x_data_oh, self.y_data = ftsepicker_lib.prepare_dataset_dectree(self.processed_dataset,
                                                            keep_cols,
                                                            y_col,
                                                            oh_cols)
        
        return

    def train_ml_predictor(self):
        return

    def make_predictions(self):
        return