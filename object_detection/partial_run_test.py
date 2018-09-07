import tensorflow as tf
from tensorflow.python import debug as tfdbg
import time

a = tf.placeholder(tf.float32, shape=[])
b = tf.placeholder(tf.float32, shape=[])
c = tf.placeholder(tf.float32, shape=[])
r1 = tf.add(a, b)
r2 = tf.multiply(r1, c)


start = time.time()
sess = tf.Session()
h = sess.partial_run_setup([r1, r2], [a, b, c])

sess = tfdbg.LocalCLIDebugWrapperSession(sess)

res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
res = sess.partial_run(h, r2, feed_dict={c: 2})
# res = sess.partial_run(h, r2, feed_dict={c: 3})

print(res)
end = time.time()
print(str((end - start)*1000) + ' ms')