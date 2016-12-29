from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

# ----------------------------

import tensorflow as tf

learning_rate = .001
training_epochs = 15
batch_size = 100
display_step = 1

n_input = 784
n_classes = 10
n_hidden_layer = 256

weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.zeros([n_hidden_layer])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
x_flat = tf.reshape(x, [-1, n_input])

layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
relu = tf.nn.relu(layer_1)
logits = tf.add(tf.matmul(relu, weights['out']), biases['out'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        num_batches = mnist.train.num_examples // batch_size
        for i in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    output = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

    save_file = "models/mnist_{0}.ckpt".format(output)
    saver.save(sess, save_file)

