# Sean R. Lang (014749564)

'''
Configuration Options
'''
import os

DEBUG = True

WUNDERGROUND_TOKEN = 'hidden'
WUNDERGROUND_URL_NOW = 'http://api.wunderground.com/api/%s/geolookup/conditions/q/NY/New_York.json' % WUNDERGROUND_TOKEN

_DEV_SQL_URI = 'sqlite:///tmp/db.sqlite3'
_GOOG_SQL_URI = 'google cloud database uri'
_DEV_HOST = 'hidden'
_DEV_MACH = DEBUG and os.uname()[1] == _DEV_HOST
SQLALCHEMY_DATABASE_URI  = _DEV_SQL_URI if _DEV_MACH else _GOOG_SQL_URI
