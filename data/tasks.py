# Sean R. Lang (014749564)

'''
Tasks to be used with Google Cloud's PubSub task management system
for the sake of updating data and the machine learning model.

---- was never utilized ----
'''

import requests

from config import WUNDERGROUND_URL_NOW
from model import database as db, Stock, Weather


def query_weather_now(date):
    r = requests.get(WUNDERGROUND_URL_NOW)

def query_stocks_now():
    pars = {}
    r = requests.get(FINANCE_API_NOW, params=pars)


def add_weather():
    to_add = Weather() # TODO
    db.session.add(to_add)
    db.session.commit()
    pass

def update_weather(date, **kwargs):
    upd = Weather.query.filter_by(date=date).first()
    pass

def add_stock(**kwargs):
    '''
    kwargs = dict{
        'date': str/date
        'ticker': str
        'p_open': float
        'p_close': float
        'volume': float
    }
    '''
    to_add = Stock(date=kwargs['date'],
                    ticker=stock_date['ticker'],
                    p_open=kwargs['p_open'],
                    p_close=kwargs['p_close'],
                    volume=kwargs['volume'])
    db.session.add(to_add)
    db.session.commit()

def update_stock(date, ticker, **kwargs):
    upd = Stock.query.filter_by(date=date).filter_by(ticker=ticker)[0]
    # TODO: update values
    # TODO: commit changes
    pass
