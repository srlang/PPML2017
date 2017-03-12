#!/usr/bin/env python

# Sean R. Lang (014749564)

''' Using Quandl to download stock information about the DJIA '''

#from pprint import pprint
import quandl
import datetime

big_start_date = '2011-01-01'
start_date = '2017-02-15'
auth_token = 'hidden'
today = datetime.datetime.now().strftime('%Y-%m-%d')
print(today)
q = quandl.get("GOOG/INDEXDJX_DJI",
                authtoken=auth_token,
                start_date=start_date,
                end_date=today)

print(q)

print(q[start_date]['Date'])


