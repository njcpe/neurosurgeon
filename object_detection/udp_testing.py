# Sample UDP Server - Multi threaded
import zstandard as zstd
import base64 as b64
import csv
import io
import os
import queue
import sys
import threading
import time
from multiprocessing import Queue, Pool,Process
from socketserver import DatagramRequestHandler, ThreadingUDPServer
import zlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread

from object_detector import PARTITION_NAME


import object_detector
import ujson

from utils.classification_utils import Frame
from utils.network_utils import RequestType, generateErrorMsg, generateResponse, generateResultMsg

from preprocessor import PreprocessorWorker

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


controlPort = 9998 

dataList = []
numRECV = 0
numSlices = -1
inputShape = (1, 9216)
mobileProcTime = 0
transmitStartTime = 0
endToEndStartTime = 0
DTYPE = "JACKSON"

ServerAddress = ('', controlPort)

def postProcess(socket, address, port, output_queue, numPictures):
    print("postProcessor Started")
    print(numPictures)

    while True:
        results = []
        for x in range(numPictures):
            # print(x)
            results.append(output_queue.get())
            # print("hello")
        stopTime = time.time()

        # print('test')
        for frame in results:
            frame.stopServerProcTimer(stopTime)
            socket.sendto(generateResultMsg(frame),(address, port))
            print('sent')
        print("done sending results")


# Subclass the DatagramRequestHandler
class ControlMixin(object):
    def __init__(self, handler, poll_interval):
        self._thread = None
        self.poll_interval = poll_interval
        self._handler = handler

    def start(self):
        self._thread = t = threading.Thread(target=self.serve_forever,
                                            args=(self.poll_interval,))
        t.setDaemon(True)
        t.start()
        # print("server running")

    def stop(self):
        self.shutdown()
        self._thread.join()
        self._thread = None


        
class EasyUDPServer(ControlMixin, ThreadingUDPServer):
    def __init__(self, input_queue, output_queue, addr, handler, poll_interval=0.001, bind_and_activate=True):
        self.input_queue = input_queue
        self.output_queue = output_queue
        #self.plot = plot
        #self.fig = fig
        # print("server init called")
    
        

        class MyUDPRequestHandler(DatagramRequestHandler):

            def __init__(self, request, client_address, server):
                self.input_queue = server.input_queue
                self.output_queue = server.output_queue
                self.inputShape = server.inputShape
                #self.plot = server.plot
                #self.fig = server.fig
                DatagramRequestHandler.__init__(self, request, client_address, server)

            def handle(self):
                data = self.request[0].strip()
                socket = self.request[1]
                msgParts = ujson.loads(data)
                reqType = RequestType[msgParts['MessageType']]

                
                if reqType == RequestType.HELLO:
                    print("HELLO Recv'd")
                    resp = generateResponse(reqType, PARTITION_NAME)
                    print(self.client_address[0])
                    socket.sendto(resp, (self.client_address[0],controlPort))

                elif reqType == RequestType.SETUP:
                    print("SETUP Recv'd")
                    input_queue.put(msgParts)
                    resp = generateResponse(reqType, PARTITION_NAME)
                    socket.sendto(resp, (self.client_address[0], controlPort)) #Respond to setup request
                    postProcessor = Process(target=postProcess, args = (socket,self.client_address[0], controlPort,output_queue, int(bytearray(msgParts['Payload'])))) 
                    postProcessor.start()
                    

                elif reqType == RequestType.DATA_HEADER:
                    numPictures = msgParts['NumPictures']
                    input_queue.put(msgParts)
                    
                    
                elif reqType == RequestType.DATA:
                    input_queue.put(msgParts)

                    

                elif reqType == RequestType.GOODBYE:
                    print(">>disconnect request recv'd")
                    results = generateResponse(reqType, PARTITION_NAME)
                    socket.sendto(results, (self.client_address[0], controlPort))
                else:
                    print("uhoh")

        ThreadingUDPServer.__init__(self, addr, MyUDPRequestHandler, bind_and_activate)
        ControlMixin.__init__(self, handler, poll_interval) 

    def setPartitionPt(self, partitionName, partitionDict):
        if partitionName in partitionDict.keys():
            self.inputShape = tuple(partitionDict[partitionName])
            print("Partition Point set to layer " + partitionName + " with shape " + str(self.inputShape))
        else:
            print("oops")

def main():
    #Setup the queues for the process communication between the Network Handler, Preprocessor, and Classifier
    preprocessor_input_q = Queue()
    preprocessor_output_q = Queue()
    classifier_output_q = Queue()

    
    object_detector.readPartitionData()


    udpserver = EasyUDPServer(preprocessor_input_q, classifier_output_q, ServerAddress, 0.01)
    udpserver.start()
    udpserver.setPartitionPt(PARTITION_NAME, object_detector.partitions_dict)



    try:

        preprocessor_pool = Pool(1,PreprocessorWorker,(preprocessor_input_q,preprocessor_output_q, udpserver.inputShape))
        classifier_pool = Pool(1, object_detector.worker, (preprocessor_output_q, classifier_output_q))


        udpserver.serve_forever()
    except KeyboardInterrupt:
        preprocessor_pool.terminate()
        classifier_pool.terminate()
        
        preprocessor_pool.join()
        classifier_pool.join()
        print("exiting")
if __name__ == '__main__':   
    main()
