import tensorflow as tf
from graphviz import Digraph

def tf_to_dot(graph):
    dot = Digraph()

    for n in graph.as_graph_def().node:
        dot.node(n.name, label=n.name)

        for i in n.input:
            dot.edge(i, n.name)
    
    return dot

if __name__ == '__main__':
    classification_graph = tf.Graph()

    with classification_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    tf.profiler.profile(graph=tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation()``)

    # gr = tf_to_dot(classification_graph)
    # tf.summary.FileWriter('logs',classification_graph).close()