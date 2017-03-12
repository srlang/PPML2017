# Sean R. Lang (014749564)

'''
Class to deal with creating the basic web page/form of the application to
select a date and display a prediction for it.
'''

from config import DATE_FORMAT

from flask_wtf import FlaskForm as Form
from wtforms import validators
from wtforms.fields.html5 import DateField

class DateForm(Form):
    date = DateField('Prediction Day', [validators.Required('Please input day for prediction.')], format=DATE_FORMAT)
