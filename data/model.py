# Sean R. Lang (014749564)

'''
Database model to be used for the web application.
Was not ever put into use, however, so data fields here and in the machine
learning model will differ.
'''

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from config import SQLALCHEMY_DATABASE_URI

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
#'sqlite:///tmp/db.sqlite3'

database = _db = SQLAlchemy(app)

class Weather(_db.Model):
    ''' classt/table to keep track of weather data for each day '''
    __tablename__ = 'weather'

    id = _db.Column(_db.Integer, primary_key=True)
    date = _db.Column(_db.Date, unique=True)
    temp = _db.Column(_db.Float)
    wind_dir = _db.Column(_db.Float)
    wind_spd = _db.Column(_db.Float)
    wind_gst = _db.Column(_db.Float)
    precipitation = _db.Column(_db.Float)

class Stock(_db.Model):
    ''' class/table to keep track of stock information for each day '''

    __tablename__ = 'stock'

    id = _db.Column(_db.Integer, primary_key=True)
    date = _db.Column(_db.Date)
    ticker = _db.Column(_db.String(10))
    p_open = _db.Column(_db.Float)
    p_close = _db.Column(_db.Float)
    volume = _db.Column(_db.Float)

    @property
    def p_delta():
        return p_close - p_open

class Temp(_db.Model):
    '''
    Temporary class to be used for the testing of filtering and updating:
        basically, how does filtering work and what sort of steps must
        be taken to make sure changes are saved 
    '''
    __tablename__ = '_tmp_'

    id = _db.Column(_db.Integer, primary_key=True)
    txt = _db.Column(_db.String(200))
    num = _db.Column(_db.Float)

'''
tags = _db.Table('tags',
        _db.Column(
'''
