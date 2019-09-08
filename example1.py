import tensorflow as tf
import numpy as np


def Difference(base, value):
    '''
    return the difference from value to base in percentage
    '''
    return (abs(float(base - value)) / float(base))


# create data
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure data
weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer() # !IMPORTANT!

# create session and start the training
sess = tf.compat.v1.Session()
sess.run(init)

count = 0

while Difference(0.1, sess.run(weights)) > 0.0001 or Difference(0.3, sess.run(biases)) > 0.0001:
    sess.run(train)
    count += 1
    if count % 10 == 0:
        print(count, sess.run(weights), sess.run(biases))

print("Finally:")
print(count, sess.run(weights), sess.run(biases))
