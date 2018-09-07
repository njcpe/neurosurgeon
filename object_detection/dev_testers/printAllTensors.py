import tensorflow as tf
from tensorflow.python.platform import gfile

x = tf.placeholder(tf.float32,shape=(1,224,224,3))

with tf.Session() as sess:
    model_filename = '/home/noah/tensorflow_builds/tf_cpu/tensorflow/tensorflow/examples/android/assets/tensorflow_inception_graph.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

    for op in tf.get_default_graph().get_operations():
        print(str(op.values()))