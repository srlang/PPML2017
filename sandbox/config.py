# Sean R. Lang (014749564)

''' Configuration options '''

# Data files
NASDAQ_FILE = '../data/stocks/hist_nasdaq.csv'
DOW_FILE = '../data/stocks/hist_dow.csv'


# weather files may be a single file name, or a list/tuple of name strings
WEATHER_FILES = [
        '../data/weather/hist_data_1.csv',
        '../data/weather/hist_data_2.csv',
        ]

### Date format specifiers
DATE_FORMAT_WEATHER = '%Y%m%d'
DATE_FORMAT_NASDAQ = '%Y-%m-%d'
DATE_FORMAT_DOW = '%Y/%m/%d'

### Factors to/for prediction
CONTINUOUS_FACTORS = [
        'tavg',
        'tmin',
        'tmax',
        'prcp',
        'snwd',
        'snow',
#        'awnd',
#        'wdf2',
#        'wsf2',
#        'wt01',
#        'wt06',
#        'wt02',
#        'wt04',
#        'wt08',
#        'open',
        ]
CATEGORICAL_FACTORS = [
        #'ticker',
        ]
PREDICT_COLUMN='delta'
