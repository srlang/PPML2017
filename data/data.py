# Sean R. Lang

'''
Basic operations to retrieve data

Never got beyond the outlining phase because the model fell through the 
floor.
'''

from json import loads as jloads
from requests import get as rget

WEATHER_TOKEN = 'hidden'
WEATHER_NOW_URL = 'http://api.wunderground.com/api/%s/geolookup/conditions/q/NY/New_York.json' % WEATHER_TOKEN
NON_FLOAT = -9999

def f(s):
    ''' convert string to float with default value if failure occurs '''
    # Unused
    try:
        return float(s)
    except ValueError:
        return NON_FLOAT


def get_weather_now():
    '''
    retrieve current weather conditions for New York, NY as of the time
    of request.
    '''
    r = rget(WEATHER_NOW_URL)
    js = jloads(r.text)
    curr_weather = js['current_observations']
    ret = {}
    ret['weather_str'] = curr_weather['weather']
    ret['wind_chl'] = float(curr_weather['windchill_c'])
    ret['wind_dir'] = float(curr_weather['wind_degrees'])
    ret['wind_spd'] = float(curr_weather['wind_kph'])
    ret['wind_gst'] = float(curr_weather['wind_gust_kph'])
    ret['temp_c'] = float(curr_weather['temp_c'])
    ret['feels_like_c'] = float(curr_weather['feelslike_c'])
    ret['heat_index_c'] = float(curr_weather['heat_index_c'])
    ret['uv'] = float(curr_weather['UV'])
    ret['pressure_mb'] = float(curr_weather['pressure_mb'])
    ret['precipitation'] = float(curr_weather['precip_today_metric'])
    ret['solar_rad'] = float(curr_weather['solarradiation'])
    return ret

def get_weather_forecast(days=10):
    ''' retrieve the weather forecast for New York, NY '''
    r = rget(WEATHER_FORECAST_URL)
    



