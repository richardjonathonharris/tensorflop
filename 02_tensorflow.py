import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.ion()
n_observations = 1000
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
ax.scatter(xs, ys)
plt.draw()
plt.savefig('02_plot1.png')

# setting tensorflow placeholders to get ready for data

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# W is just the weight on the "linear regression"

W = tf.Variable(tf.random_normal([1]), name='weight')

# B is bias, which is just the constant

b = tf.Variable(tf.random_normal([1]), name='bias')

# and this is how we compute Yhat

Y_pred = tf.add(tf.multiply(X, W), b)

# Here's loss function that we try to minimize over epochs
# Obvi it's the SSE

cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

# We're gonna use Gradient Descent to minimize the cost function

learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

## Now let's train this model yo!

n_epochs = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # initializes all variables in graph
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(
                cost, feed_dict={X: xs, Y: ys})
        print training_cost

        if epoch_i % 20 == 0:
            ax.plot(xs, Y_pred.eval(
                feed_dict={X: xs}, session=sess),
                'k', alpha=epoch_i / n_epochs, color='r')
            plt.draw()

        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost

plt.savefig('02_plotplot.png')
