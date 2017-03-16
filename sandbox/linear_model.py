# Sean R. Lang (014749564)

''' Code to create a linear model for the sake of debugging things. '''

import os

import tensorflow as tf
import pandas as pd

from read_data import load_data

# TODO: more columns eventually for more complexity later (maybe)
# NB. make sure that delta is the last one for simple use later
#       (things later rely upon filtering out the predicted value as the
#        last item)
FEATURE_LABELS = ['prcp', 'snow', 'tavg', 'tmax', 'tmin', 'delta']

CHECKPOINT_FILE = '/tmp/tensorflow/model.ckpt'

def create_model():
    '''
    Create the model from basic tensors.
    Returns all information/tensors necessary to manipulate later.
    '''
    #tmin = tf.placeholder(tf.float32)
    #tmax = tf.placeholder(tf.float32)
    #tavg = tf.placeholder(tf.float32)
    #snow = tf.placeholder(tf.float32)
    #prcp = tf.placeholder(tf.float32)
    #ticker = tf.contrib.layers.sparse_column_with_hash_bucket(\
    #            'ticker', hash_bucket_size=100)
    features_len = len(FEATURE_LABELS) - 1

    # input placeholder
    x = tf.placeholder(tf.float32, shape=[features_len, 1])
            #, shape=[None,None])
    # placeholder for the actual output (for training)
    y_actual = tf.placeholder(tf.float32) #, shape=[None, None])

    # weights coefficient matrix
    #W = tf.Variable(tf.zeros([features_len, features_len]), name='Weight')
    W = tf.Variable(tf.zeros([1, features_len]), name='Weight')
    # bias coefficient
    #b = tf.Variable(tf.zeros([features_len]), name='bias')
    b = tf.Variable(tf.zeros([1]), name='bias')

    #x = [tmin, tmax, tavg, snow, prcp] #, ticker]

    # The fact that even a PRINT STATEMENT won't work, means I've exhausted
    #   my capacity to figure out what the hell is causing this to fail.
    # That's correct, the following two tensors do not actually give any
    #   output and there is no difference between my code and examples for
    #   this to be the case.
    #p_W = tf.Print(W, [W], 'Weight: ')
    #p_b = tf.Print(b, [b], 'bias: ')
    last_straw_W = tf.get_variable('Weight', [features_len,features_len])
    last_straw_W = tf.Print(last_straw_W, [last_straw_W], message='Please:')
    # ^ still didn't work, but magically, the NaN issues disappeared and
    #   were replaced with errors and predictions  on the order of 1e14
    #   where actual answer should be on the order of 1e-2 (predicting
    #   fractional change rather than percentage directly as it is easy
    #   enough to change when displaying and it makes thing simpler
    #_y_tmp = tf.matmul(x, W, transpose_a=True) + b
    _y_tmp = tf.matmul(W, x) + b
    p_y_tmp = tf.Print(_y_tmp, [_y_tmp], 'Wx + b=', first_n=100)
    y_pred = tf.reduce_sum(tf.matmul(x, _y_tmp), name='y_pred') 
    # transpose_a=True)
 
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #        labels=y_actual, logits=y_pred))
    #cost = tf.reduce_sum(tf.pow(y_pred-y_actual, 2))/(2*features_len)
    #cost = tf.reduce_sum(tf.pow(tf.subtract(y_actual, y_pred), 2))
    #cost = tf.reduce_sum(tf.abs(tf.subtract(y_actual, y_pred)), name='cost')
    # used absolute value for the sake of just trying anything to get over
    #   the NaN issues.
    #   returned things back to square difference to make sure that there
    #   wasn't some stupid bug in abs()that was causing the error to grow
    #   the square method gives me my NaN issues back (yay!)
    #       --> src: squaring a huge number gets beyond range of float32
    #   Instead, we will use a sqrt error to make things smaller and see
    #   if we can find any improvement
    #   Another reason NaNs kept showing up despite earlier changes was
    #   my insistence on reloading models from previous training.
    #   This caused out-of-bounds float32s to be loaded and used (nan)
    #   and any further math on these numbers failed
    #   This project was just a testament to how much of a clusterf***
    #   I can make something.
    #   Sudden thought: probably has something to do with the fact that
    #   there are -9999.0 values in some fields (as errors)
    #       I figured that these would just receive low weights, but now
    #       it looks like a good idea would be to filter and replace those
    #       with 0s instead
    #   This "fix" turned out to improve things slightly.
    #   There is still a large error that varies between 200 and 50K
    #       (at the time, this was _with_ a sqrt op too,
    #        so that's REALLY bad, because predicted values were still in
    #        the range of billions and we're predicting price per share
    #        which is in like the 60s/70s now)
    cost = tf.reduce_sum(tf.sqrt(tf.abs(tf.subtract(y_actual, y_pred))), name='cost')
    #cost = tf.pow(y_actual-y_pred, 2)
    learn_rate = 0.1
    #learn_rate = 25.000000  # stupidly high to see if we get anywhere
                            # close
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).\
            minimize(cost) #, var_list=[W,b])
        #.minimize(loss)
    return x, y_pred, optimizer, y_actual, cost, None

def get_train_data(df):
    ''' get a training data set by using a random sampling of the data '''
    sample = df.sample(frac=0.75)
    #return df.sample(frac=0.75).as_matrix([FEATURE_LABELS])
    return sample.as_matrix(FEATURE_LABELS[:-1]) , sample.as_matrix([FEATURE_LABELS[-1]])

def get_test_data(df):
    ''' get a test data set by using a random sampling of data '''
    sample = df.sample(frac=0.20)
    #return df.sample(frac=0.20).as_matrix([FEATURE_LABELS])
    return sample.as_matrix(FEATURE_LABELS[:-1]) , sample.as_matrix([FEATURE_LABELS[-1]])

def feed(x, y):
    ''' Helper method to assist in inputting data (did not work) '''
    return {
            'x': x,
            'y_actual': y,
            }

def force_feed(array):
    ''' Format an array into a matrix for input into the model '''
    r = []
    for x in array:
        r.append([x])
    return r

def init(session, saver, checkpoint_file=CHECKPOINT_FILE):
    '''
    Initialize a graph session with a given Saver object
    Will load a graph from a checkpoint saved on disk if one can be found
    '''
    if os.path.exists(checkpoint_file+'.meta'):
        saver.restore(session, checkpoint_file)
    else:
        init = tf.global_variables_initializer()
        session.run(init)

def save(session, saver, outfile=CHECKPOINT_FILE):
    ''' save a graph session to disk '''
    saved_path = saver.save(session, outfile)
    return saved_path


def main():
    '''
    method to be the main method so I can use it from an interpreter
    session
    Trains and evaluates a (linear) model.
    '''
    DEBUG = True
    training_sessions = 200
    # 10k now b/c i want to see how not terrible it /can/ get
    display_step = 10
    save_step = 5

    x_i, model, optimizer, y_act, cost, summary = create_model()
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(model)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    _,_,data = load_data()

    if DEBUG:
        print(data[0:4])

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init(sess, saver)

        for i in range(training_sessions+1):
            train_x, train_y = get_train_data(data)
            test_x, test_y = get_test_data(data)
            ######

            for j in range(len(train_x)):
                x_in = force_feed(train_x[j])
                y_out = train_y[j]
                sess.run(optimizer, feed_dict={x_i:x_in, y_act:y_out})
                #print(optimizer.eval())

            if i % save_step == 0:
                saved_path = save(sess, saver) 
                #saver.save(sess, CHECKPOINT_FILE)
                print('saved step %d at %s' % (i, saved_path))
            
            if i % display_step == 0:
                r = 0 # later randomize
                #loss = sess.run(cost, feed_dict=feed(test_x, test_y))
                x_in = force_feed(test_x[r])
                y_o = test_y[r]
                y_f = sess.run(model, feed_dict={x_i:x_in}) #, y_act:y_o})
                c = sess.run(cost, feed_dict={x_i:x_in, y_act:y_o})
                print('c')
                print(str(c))
                #print(c.eval())
                print('input: %s, output: %f, expected: %f, cost %f' % \
                        (test_x[r], y_f, y_o, c))
                print('%d step; cost: %s' % (i, c))
                # LaTeX table of "results" (easy enough to grep/sed out later)
                # Headers: Iteration number, expected, predicted, cost
                print('%% %d & %.5ff & %.5f & %.2f \\\\' % (i, y_o, y_f, c))
                #print(c.eval())

                # Summarize stuff
                summary = tf.summary.merge_all()
                fw = tf.summary.FileWriter('./logs', sess.graph)
            ######
            '''

            #model.fit(input_fn=train_input_fn, steps=100)
            model.fit(x=train_x, y=train_y)

            results = model.evaluate(input_fn=eval_input_fn, steps=1)
            for k in sorted(results):
                print('%s: %s' % (k, results[k]))
            '''


if __name__ == '__main__':
    main()
