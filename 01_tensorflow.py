import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

n_values = 32
x = tf.linspace(-3.0, 3.0, n_values)

print x # 32 values evenly spaced between -3.0 and 3.0

sess = tf.Session()
result = sess.run(x)

print result

sigma = 1.0
mean = 0.0

z = (tf.exp(tf.negative(tf.pow(x-mean, 2.0) /
    (2.0 * tf.pow(sigma, 2.0)))) *
    (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

plt.plot(z.eval(session=sess))
plt.savefig('01_plot.png')

print z.get_shape()
print z.get_shape().as_list()

z_2d = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))

sess = tf.InteractiveSession()
print z_2d.eval(session=sess)

plt.clf()
plt.imshow(z_2d.eval())
plt.savefig('01_secondplot.png')

ops = tf.get_default_graph().get_operations()
print [op.name for op in ops]
