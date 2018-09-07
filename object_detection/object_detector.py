import argparse
import base64
import json
import multiprocessing
import os
import time
from io import BytesIO
from multiprocessing import Pool, Queue
import sys

import cv2
import numpy as np
import tensorflow as tf
import tfgraphviz as tfg
from tensorflow.python import debug as tfdbg
from tensorflow.python.client import timeline

from utils import label_map_util
from utils import visualization_utils as vis_util
from utils.network_utils import NSCPServer

CWD_PATH = os.getcwd()
LOGDIR = '/tmp/tensorboard'

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# Use JIT Compilation to get speed up. Experimental feature.
# config = tf.ConfigProto()
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


# # Path to frozen detection graph. This is the actual model that is used for the object detection.``
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_DIR = 'frozen_models'
# PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, MODEL_NAME, 'frozen_inference_graph.pb')

# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')


# Path to frozen classification graph. This is the actual model that is used for the object classification.``
MODEL_NAME = 'tensorflow_inception_graph'
MODEL_DIR = 'frozen_models'
PARTITION_NAME = 'mixed3a'

PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'imagenet_comp_graph_label_strings.txt')
NUM_CLASSES = 90

# Loading label map
print(">>Using " + MODEL_NAME + " Inference Model")
print(">>Loading Label Map")
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
    
def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v)) for k, v in dictionary.items())

def classify_objects(input_data, sess, classification_graph):

    input_tensor = classification_graph.get_tensor_by_name(PARTITION_NAME+':0')
    classifications = classification_graph.get_tensor_by_name('output:0')

    start = time.time()
    classifications = sess.run(classifications, feed_dict={input_tensor: input_data}, options=options, run_metadata=run_metadata)
    print('>>tClassify: '+str((time.time() - start)*1000)+' ms')
    return classifications

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0) #TODO: Make if sstatemet for YOLO Here
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # masks = detection_graph.get_tensor_by_name('detection_masks:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded}, options=options, run_metadata=run_metadata)


    detection_dict = vis_util.create_detection_dict(
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8, min_score_thresh=0.5)

    detection_dict_annotated = {}
    detection_dict_annotated = [{"object_location": i, "object_data": j} for i, j in detection_dict.items()]
    detection_dict_json = json.dumps(detection_dict_annotated)
    return detection_dict_json

def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    print(">>Loading Frozen Graph")
    classification_graph = tf.Graph()
    with classification_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            # """
            # Get all ops in Graph, used for finding part points, DEBUG
            # """
            # for op in tf.get_default_graph().get_operations():
            #     print(str(op.name))

            # print([x for x in tf.get_default_graph().get_operations()])

        sess = tf.Session(graph=classification_graph)  #config enable for JIT
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    # tfg.board(detection_graph, )
    while True:
        (partitionData, src) = input_q.get()
        '''
        Returns ((classifications, src socket obj.), start time). TODO simplify maybe.
        '''
        dataBundle = (classify_objects(partitionData, sess, classification_graph), src)
        # print(dataBundle)
        output_q.put(dataBundle)
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nw', '--num-workers', dest='num_workers',
                        type=int, default=1, help='Number of workers.')
    parser.add_argument('-qs', '--queue-size', dest='queue_size',
                        type=int, default=10, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBWARNING)

    server = NSCPServer('', 4002)
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    oldData = (None,None)

    while True:
        try:
            while (server.isClientConnected and server.isClientReady):
                try:
                    input_q.put(server.read())
                except:
                    continue

                (classifications, src) = output_q.get()
                out = (classifications, src)
                # end = t.time()
                # print("Frame processing time: " + str((end - start)) + " seconds.")
                # start = time.time()
                eql = np.array_equal(out[0], oldData[0])
                # end = time.time()
                # print("equality processing time: " + str((end - start)*1000) + " ms.")
                if not eql:
                    oldData = out
                    start = time.time()
                    server.appendToMessageBuff(classifications, src)
                    end = time.time()
                    print('>>tGenAndEnqueue: ' + str((end - start) * 1000) + ' ms')
                else:
                    print('>>theres a problem')

        except KeyboardInterrupt:
            # server.shutdown()
            sys.exit()
            # try:
            #     pool.terminate()
            #     break
            # except:
            #     print('PROBLEM')
            #     sys.exit()

    print('>>joining pool')
    pool.join()
    print('>>the end')