# Sean R. Lang (014749564)

'''
This file followed the tutorial for the wide and deep neural network
on the TensorFlow site.
There was a problem with the import utility which caused me to think
the model was failing when it kept giving me 0.0% errors.

Later on, there were still errors causing a KeyError for no discernible
reason.
'''
import tensorflow as tf
import pandas as pd

import tempfile

from read_data import load_data
from config import CONTINUOUS_FACTORS, CATEGORICAL_FACTORS, PREDICT_COLUMN
## h4ck t3h pl4n3t
#CATEGORICAL_FACTORS = []
FEATURE_COLS = CONTINUOUS_FACTORS + CATEGORICAL_FACTORS 

def input_fn(df):
    ''' format a dataframe as a set of tensors for the graph '''
    continuous_cols = {k: tf.constant(df[k].values)
                        for k in CONTINUOUS_FACTORS}
#    categorical_cols = {k : tf.SparseTensor(\
#            tf.contrib.layers.sparse_column_with_hash_bucket(\
#                'ticker', hash_bucket_size=10)
    categorical_cols = {k: tf.SparseTensor(
            indices=[[i,0] for i in range(df[k].size)],
            values=df[k].values,
            shape=[df[k].size, 1])
                for k in CATEGORICAL_FACTORS}

    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    label = tf.constant(df[PREDICT_COLUMN].values)
    return feature_cols, label

def train_input_fn():
    ''' create training input tensors '''
    global dataframe
    train = dataframe.sample(frac=0.75)
    return input_fn(train)

def eval_input_fn():
    ''' create testing input tensors '''
    global dataframe
    test = dataframe.sample(frac=0.20)
    return input_fn(test)

_,_,dataframe = load_data() 
model = None
def create_model():
    '''
    create the model using tf.Learn library to create a wide-n-deep
    neural network as per tutorial
    Later this would become a simple linear regression model
    although this was not the case
    '''
    global dataframe
    global model
    mdir = tempfile.mkdtemp()
    cat_col = [
            #tf.sparse_placeholder(x)
            tf.contrib.layers.sparse_column_with_hash_bucket(\
                    x, hash_bucket_size=100)
            for x in CATEGORICAL_FACTORS]
    con_col = [
            tf.contrib.layers.real_valued_column(x)
            #tf.placeholder(x)
            for x in CONTINUOUS_FACTORS]
    all_cols = cat_col + con_col
    wide_columns = all_cols # cat_col # con_col #cat_col
    deep_columns = con_col + [
            tf.contrib.layers.embedding_column(x, dimension=8)
            for x in cat_col
            ]
    #fcols, label = input_fn(
#    model = tf.contrib.learn.DNNLinearCombinedClassifier(
#            #loss=tf.contrib.losses.mean_squared_error,
#            model_dir=mdir,
#            linear_feature_columns=wide_columns,
#            dnn_feature_columns=deep_columns,
#            dnn_hidden_units=[100, 50])
            #BetterClassifier(
            #LinearClassifier(feature_columns=feature_cols
    model = tf.contrib.learn.LinearRegressor(\
                    feature_columns=all_cols,
                    model_dir=mdir,
                    optimizer=tf.train.FtrlOptimizer(
                        learning_rate=0.1,
                        l1_regularization_strength=1.0,
                        l2_regularization_strength=1.0
                        )
                    )
            #tf.contrib.learn.SKCompat(\
             #   )
    #m.fit(input_fn=trai
#    model.fit(input_fn=train_input_fn, steps=1000)
#    results = model.evaluate(input_fn=eval_input_fn, steps=1)
#    for key in sorted(results):
#        print('%s: %s' % (key, results[key]))

    return model


if __name__ == '__main__':
    #global dataframe
    print('continuous_factors: ')
    for x in CONTINUOUS_FACTORS:
        print(x)
    print('categorical_factors: ')
    for x in CATEGORICAL_FACTORS:
        print(x)

    m = create_model()
    for i in [1]: #range(1,2): #10):
        m.fit(input_fn=train_input_fn, steps=1)
        results = m.evaluate(input_fn=eval_input_fn, steps=1)
        for k in sorted(results):
            print('%s: %s' % (k, results[k]))
        #prediction = m.predict(eval_input_fn())
        _r = 0
        _to_predict = dataframe[FEATURE_COLS].iloc[[_r]]
        #.as_matrix(FEATURE_COLS)
        #_to_predict,_ = input_fn(dataframe.iloc[[_r]])
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('_to_predict:')
        print(_to_predict)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
        prediction = m.predict(_to_predict) #, as_iterable=False)
        #, dtype=float)
        print('prediction: %s' % str(prediction))

    #print('testing')
    #for i in range(1,2)
