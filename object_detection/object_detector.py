import argparse
import base64
import json
import multiprocessing
import os
import sys
import time
from multiprocessing import Pool, Queue

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tfdbg
from tensorflow.python.client import timeline

from utils import label_map_util
from utils import visualization_utils as vis_util

from utils.caffe_classes import class_names
from utils.network_utils import NSCPServer
from tensorflow.python.client import timeline

# from udp_testing import EasyUDPServer, MyUDPRequestHandler

CWD_PATH = os.getcwd()
LOGDIR = '/tmp/tensorboard'

HALT_SIGNAL = False

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Use JIT Compilation to get speed up. Experimental feature.
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1


# # Path to frozen detection graph. This is the actual model that is used for the object detection.``
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_DIR = 'frozen_models'
# PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, MODEL_NAME, 'frozen_inference_graph.pb')
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')


# Path to frozen classification graph. This is the actual model that is used for the object classification.``
MODEL_NAME = 'alexnet'
MODEL_DIR = 'frozen_models'

PARTITION_NAME = 'Placeholder'

OUTPUT_NAME = 'Softmax'


PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, MODEL_NAME, 'alexnet_frozen.pb')
PATH_TO_PARTN = os.path.join(CWD_PATH, MODEL_DIR, MODEL_NAME)
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'imagenet_comp_graph_label_strings.txt')
NUM_CLASSES = 90


partitions_dict = {}


# Loading label map
# print(">>Using " + MODEL_NAME + " Inference Model")
# print(">>Loading Label Map")
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
    
def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v)) for k, v in dictionary.items())

def classify_objects(frame, sess, classification_graph):
    
    input_data = frame.getImageData()

    with tf.device('/device:GPU:0'):
        input_tensor = classification_graph.get_tensor_by_name(PARTITION_NAME + ':0')
        classifications = classification_graph.get_tensor_by_name(OUTPUT_NAME + ':0')

        classifications = sess.run(classifications, feed_dict={input_tensor: input_data}, options=options, run_metadata=run_metadata)
        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # with open('timeline_01.json', 'w') as f:
        #     f.write(chrome_trace).
        frame.deleteRawImgData()
        
        ind = np.argpartition(classifications[0], -3)[-3:]
        sorted_ind = ind[np.argsort(classifications[0,ind])]
        frame.detected_objects = [class_names[indx] for indx in sorted_ind] 
        frame.confidences = (classifications[0, sorted_ind]).tolist()
        print(frame.detected_objects)
        print(frame.confidences)
    return frame

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


def worker(input_q, output_q,):

    # Load a (frozen) Tensorflow model into memory.
    # print(">>Loading Frozen Graph")
    with tf.device('/device:GPU:0'):        
        classification_graph = tf.Graph()
        with classification_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.FastGFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=classification_graph, config=config)  #config enable for JIT
    
    try:
        while 1:
            try:
                frame = input_q.get_nowait()
                '''
                Returns Frame object
                '''
                output_q.put(classify_objects(frame, sess, classification_graph))
            except:
                continue
    except KeyboardInterrupt:
        print("closing session...")
        sess.close()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nw', '--num-workers', dest='num_workers',
                        type=int, default=1, help='Number of workers.')
    parser.add_argument('-qs', '--queue-size', dest='queue_size',
                        type=int, default=10, help='Size of the queue.')
    args = parser.parse_args()

    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG)
    
    """
    Read In partition data from txt file (CSV-type format)
    """

    with open(os.path.join(PATH_TO_PARTN, 'alexnet_partitions.txt'), 'r') as f:
        
        partitions = f.readlines()
        partitions = [x.strip().split(',') for x in partitions]

        for x in partitions:
            partitions_dict[x[0]] = x[1:len(x)]

        for x, y in partitions_dict.items():
            partitions_dict[x] = list(map(int, y))
                
        for x, y in partitions_dict.items():
            # print(x,y)
            pass
    
    ServerAddress = ('', 9998)
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    server = EasyUDPServer(input_queue=input_q, output_queue=output_q, handler=MyUDPRequestHandler, addr=ServerAddress, poll_interval=0.001)

    pool = Pool(args.num_workers, worker, (input_q, output_q))

    # print('>>setting partition point')
    # server.setPartitionPt(PARTITION_NAME, partitions_dict)

    server.start()
    
    input("hit enter to stop")
    # oldData = (None,None)
    #try:  
    #    while True:
    #        try:
    #            print(server.read())
    #        except:
    #            continue

    #except KeyboardInterrupt:
    #    server.stop()
    #    sys.exit()

    #     try:
    #         while (True):
    #             try:
    #                 # input_q.put(server.read())
    #                 # print(input_q.qsize())
    #                 pass
    #             except:
    #                 continue

    #             # frame_obj = output_q.get()
    #             # server.appendToMessageBuff(frame_obj)


    # print('>>joining pool')
    # pool.join()
    # print('>>the end')
