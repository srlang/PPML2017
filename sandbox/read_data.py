# Sean R. Lang

'''
Majorly important file to read in data and format into a dataframe for
manipulation by the machine learning model.
'''

import tensorflow as tf
import pandas as pd
import numpy as np

from config import NASDAQ_FILE, DOW_FILE, WEATHER_FILES,\
        DATE_FORMAT_WEATHER, DATE_FORMAT_NASDAQ, DATE_FORMAT_DOW,\
        CONTINUOUS_FACTORS

def format_df(df, date_format='%Y-%m-%d'):
    ''' format a dataframe to make it consistent and useful '''
    df.columns = [x.lower() for x in df.columns]
    df['date'] = df['date'].apply(\
            lambda x: pd.to_datetime(str(x), format=date_format))
    return df

def post_format(df):
    ''' format a dataframe further after a join operation '''
    ERR_VAL = 999.0 # beyond this, it's usually a missing value
    #   NOTE: this will fail if volume of trading is ever taken into account
    df = df.fillna(9999.0) #.astype(np.float32)
    for factor in CONTINUOUS_FACTORS:
        #df[df[factor >= ERR_VAL or factor <= -ERR_VAL]][factor] = 0.0
        df[factor] = df[factor].apply(\
                lambda x: x if x <= ERR_VAL and x >= -ERR_VAL else 0)
    #for col in CONTINUOUS_COLS:
    # force 32-bit precision floats
    #   (want to make sure this isn't the cause of my NaN issues)
    df[CONTINUOUS_FACTORS] = df[CONTINUOUS_FACTORS].astype('float32')
    return df



DATE_FORMAT_WEATHER = '%Y%m%d'
def read_multiple_weather_csvs(weather_files):
    '''
    read multiple weather files and join them together into one joint
    weather dataframe
    '''
    if type(weather_files) == type('string'):
        return format_df(pd.read_csv(weather_file), DATE_FORMAT_WEATHER)
    elif len(weather_files) == 1:
        return format_df(pd.read_csv(weather_file[0], DATE_FORMAT_WEATHER))
    else:
        f,rest = weather_files[0],weather_files[1:]
        df = format_df(pd.read_csv(f), date_format=DATE_FORMAT_WEATHER)
        for _f in rest:
            _df = format_df(pd.read_csv(_f), date_format=DATE_FORMAT_WEATHER)
            df = pd.concat([df,_df], join='inner')
        return df
            


def load_data():
    '''
    The main method for loading data from csv files.
    Loads stock and weather and combine them into a single dataframe.
    '''
    nasdaq = format_df(pd.read_csv(NASDAQ_FILE), date_format=DATE_FORMAT_NASDAQ)
    nasdaq['ticker'] = 'NDAQ'
    #dow = format_df(pd.read_csv(DOW_FILE), date_format=DATE_FORMAT_DOW)
    #dow['ticker'] = '.DJI'
    #stocks = pd.concat([nasdaq,dow], join='inner')
    stocks = nasdaq # only using nasdaq for now because we need to simplify
    # PERCENTAGE change = change over day / starting price
    #   want to predict percentage because yeah
    #   technically, this is 1/100th of the percentage (later format)
    stocks['delta'] = (stocks['close'] - stocks['open']) / stocks['open']

    weather = read_multiple_weather_csvs(WEATHER_FILES)

    data = weather.merge(stocks, on='date')
    data = post_format(data)
    return stocks,weather,data


if __name__ == '__main__':
    ''' main method tests data import '''
    # stocks, weather, data(all)
    s,w,d = load_data()
    print(d)
    print(d.columns)
    #for i in d.iterrows():
    #    print(i)
    dm = d.as_matrix()
    for i in range(len(dm)):
        #lenprint(d.iterrows()[i])
        print(dm[i])
