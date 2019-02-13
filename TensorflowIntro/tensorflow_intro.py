import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# If you are running in Ipython clear the default graph
tf.reset_default_graph()

# ----------------------------------------------------------------------------------------------------
# Generate Data
# y = mx + b
# let's make some training data
SLOPE = 2
OFFSET = -5
NUMBER_EPOCHS = 500

X = np.random.uniform(low=0,high=10, size=(100,1))
y = X*SLOPE + OFFSET

# ----------------------------------------------------------------------------------------------------
# Create Model

with tf.name_scope("Model"):
    # Now we have a line we would want to have fit.
    m = tf.Variable(0.0, dtype=tf.float32, name='m')
    b = tf.Variable(0.0, dtype=tf.float32, name='b')
    tf_y = m*X + b
    tf.summary.scalar('m', m)
    tf.summary.scalar('b', b)

with tf.name_scope("MeanSquaredError"):
    mse = tf.reduce_mean(tf.square(tf_y - y), name="mse")
    tf.summary.scalar('mse', mse)

with tf.name_scope("GradientDescent"):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    training_op = optimizer.minimize(mse)


# ----------------------------------------------------------------------------------------------------
# train the model and view results.

init = tf.global_variables_initializer()  # Tell tensorflow to Initialize athe variables (m and b)
file_saver = tf.summary.FileWriter('./model/', tf.get_default_graph())  # async file saving object
merged_summaries = tf.summary.merge_all()  # take all the tf.summary... and merge them into one

with tf.Session() as sess:  #create a session to so you can train your model
    sess.run(init)  # run all the variables you initialized in the current session

    for epoch in range(NUMBER_EPOCHS):
        # when you run these parts of the graph Tensorflow workes through the graph
        # that has been created and generates outputs.  In this case we are asking
        # for the mse, variable, an update of the summary scalars, and running the
        # gradient descent for one iteration
        error, summaries, _ = sess.run([mse, merged_summaries, training_op])

        file_saver.add_summary(summaries, epoch)

        #  Save a graph of expected vs actuall
        if epoch % 10 == 0:
            y_pred = tf_y.eval()
            print("Epoch: {:n}, Mean Squared Error: {:n}".format(epoch, error))
            plt.scatter(y_pred,X, label='Regression Prediction')
            plt.scatter(y,X, label='Expected Value')
            plt.legend()
            plt.savefig("./GradientDescentVisual/Epoch-"+str(epoch))
            plt.clf()
    
    #push any remaining data to disk
    file_saver.flush()
