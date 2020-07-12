
import pandas as pd
import requests
from . import ftsepicker_lib
from . import model_calibration
import os
import time
import io

class ProcessEngine:
    def __init__(self):
        self.model=model_calibration.ModelCalibration()
        self.download_folder=None
        return None

    def set_download_folder(self, download_folder):
        self.download_folder=download_folder
        return

    def write_to_folder(self, df):
        df.to_csv(os.path.join(self.download_folder, df))
        return

    def set_stock_universe(self, stock_list):
        self.stock_universe=stock_list
        return None
    
    
    def load_timeseries_data(self, file_path):
        self.timesseries_data=pd.read_csv(file_path)
        return None
    
    def urlBuilder(self,ticker, api_key, size='full'):
        url=r'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={1}&outputsize={2}&apikey={0}&datatype=csv'.format(api_key,ticker, size)
        return url

    def download_data(self,api_key, download_size='compact'):

        summary_df=pd.DataFrame()
        self.fail_list=[]
        for t in self.stock_universe:
            try:
                url=self.urlBuilder(t, api_key, download_size)
                r=requests.get(url)
                time.sleep(12)
                df=pd.read_csv(io.StringIO(r.content.decode('utf-8')))
                if df.shape[1]==6:
                    df['Stock']=t
                    summary_df=pd.concat([summary_df, df], axis=0)
                else:
                    print('failed to download {0}'.format(t))
                    print(df)
                    print(url)
                    self.fail_list.append(t)
            except:
                print('error occured {0}'.format(t))
                self.fail_list.append(t)
            
            self.timesseries_data=summary_df
        return

    def process_data(self):
        data_df_processed=pd.DataFrame()
        ts_data=self.timesseries_data.copy()
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
                print(data_df)
            
            self.processed_dataset=data_df_processed

            
        return None
    
    def prepare_data_for_model(self, keep_cols, y_col, oh_cols):
        
        self.x_data, self.x_data_oh, self.y_data = ftsepicker_lib.prepare_dataset_dectree(self.processed_dataset,
                                                            keep_cols,
                                                            y_col,
                                                            oh_cols)
        
        return None

    def train_ml_predictor(self):
        self.model.train_and_predict(self.x_data, self.y_data)
        return None

    def make_predictions(self, x_data, y_data):
        if self.model.is_calibrated==False:
            print('Must train model first')
            return None
        
        predictions=self.model.predict_with_model(x_data)

        return predictions