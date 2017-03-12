#!/usr/bin/env python

# Sean R. Lang

''' Weather downloading example using the wunderground api '''

# Weather example

#import urllib2
#import json
#f = urllib2.urlopen('http://api.wunderground.com/api/<token>/geolookup/conditions/q/NY/New_York.json')
#json_string = f.read()
#parsed_json = json.loads(json_string)
#location = parsed_json['location']['city']
#temp_f = parsed_json['current_observation']['temp_f']
#print("Current temperature in %s is: %s" % (location, temp_f))
#f.close()

import requests
import json 
from json import loads, dumps

from pprint import PrettyPrinter

token = 'hidden'
url = 'http://api.wunderground.com/api/%s/geolookup/conditions/q/NY/New_York.json' % token
r = requests.get(url)

#print(loads(r.text), indent=4, separators=(',', ': '))
js = loads(r.text)

def f(s):
    try:
        return float(s)
    except ValueError:
        return -9999

curr_weather = js['current_observation']
ret = {}
ret['weather_str'] = curr_weather['weather']
ret['wind_chl'] = f(curr_weather['windchill_c'])
ret['wind_dir'] = f(curr_weather['wind_degrees'])
ret['wind_spd'] = f(curr_weather['wind_kph'])
ret['wind_gst'] = f(curr_weather['wind_gust_kph'])
ret['temp_c'] = f(curr_weather['temp_c'])
ret['feels_like_c'] = f(curr_weather['feelslike_c'])
ret['heat_index_c'] = f(curr_weather['heat_index_c'])
ret['uv'] = f(curr_weather['UV'])
ret['pressure_mb'] = f(curr_weather['pressure_mb'])
ret['precipitation'] = f(curr_weather['precip_today_metric'])
ret['solar_rad'] = f(curr_weather['solarradiation'])
out = ret
#print(out, sort_keys=True, indent=4, separators=(',', ': '))
pp = PrettyPrinter(indent=4)
pp.pprint(out)
