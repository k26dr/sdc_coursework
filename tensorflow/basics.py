import tensorflow as tf

#x = tf.add (5,2)
#x = tf.sub(5,2)
#x = tf.mult(3,4)
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    feed_dict = { 
        x: "Hello World",
        y: 123,
        z: 45.67
    }
    output = sess.run(x, feed_dict=feed_dict)
    print(output)
