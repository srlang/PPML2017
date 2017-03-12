# Sean R. Lang (014749564)

'''
Main web site file.
Creates a mapping of pages to routes and allows for templates to be rendered.
(Flask is super simple for basic stuff)
'''

from flask import Flask, render_template, redirect, url_for
from datetime import datetime
#from werkzeug.routing import Ba
import json

from forms import DateForm

from config import DEBUG, DATE_FORMAT

app = Flask(__name__)
app.secret_key='hidden'

@app.route('/', methods=['POST','GET'])
def index():
    ''' index page to display the date selection form '''
    form = DateForm()
    if form.validate_on_submit():
        date = form.date.data.strftime(DATE_FORMAT)
        return redirect(url_for('view_prediction', pred_date=date))
    return render_template('date_picker.html', form=form)


@app.route('/predict/<pred_date>', methods=['POST','GET'])
def view_prediction(pred_date):
    ''' view a prediction for a given date '''
    date = datetime.strptime(pred_date, DATE_FORMAT)
    return render_template('prediction_view.html', date=date)


@app.route('/print/<s>/', methods=['POST','GET'])
def print_api(s):
    ''' testing just to see what would happen '''
    return s


if __name__ == '__main__':
    app.run(debug=DEBUG)
