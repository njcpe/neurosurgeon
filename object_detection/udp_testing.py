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
from enum import Enum
from multiprocessing import Pool, Queue
from socketserver import DatagramRequestHandler, ThreadingUDPServer
from sys import getsizeof
import zlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread

import object_detector
import ujson
from concurrent import futures

from utils.classification_utils import Frame


flatten = lambda l: [item for sublist in l for item in sublist]
flattenDecode = lambda l: [n for sublist in l for n in b64.b64decode(sublist)]


class RequestType(Enum):
    HELLO = 'OK'
    SETUP = 'SETUP'
    READY_ACK = 'READY_ACK'
    PLAY = 'ACK'
    DATA_HEADER = 'DATA_HEADER_RECV'
    DATA = 'DATA_RECV'
    RESULTS = 6
    GOODBYE = 'GOODBYE'

def generateResponse(requestType):
        # print('TESTING: ' + requestType.value)
        body = {}
        body_str = ""
        content_len = 0

        if (requestType == RequestType.SETUP):  #TODO: Add partition logic
            
            body.update({'Partition-Point': object_detector.PARTITION_NAME})
            body.update({'UDP-Ports': [9998,9999]})
            
            # print(body)
            body_str = ujson.dumps(body)
            # print(body_str)
            content_len = getsizeof(body_str.encode())

        elif (requestType == RequestType.GOODBYE):
            body_str = ""
            content_len = getsizeof(body_str.encode())

            
        response = requestType.value + '\r\n' + \
                    'CSeq: ' + str(0) + '\r\n' + \
                    'Content-Length: ' + str(content_len) + '\r\n' + \
                    body_str + '\r\n\r\n'
        return response.encode()

def generateErrorMsg():
    # frame.stopServerProcTimer()
    # print('TESTING: ' + requestType.value)
    body = {}
    body_str = ""
    content_len = 0

    #Add metadata
    body.update({'mobileProcDeltaTime': -1})
    body.update({'serverProcDeltaTime': -1})
    body.update({'transmitStartTime': -1})
    body.update({'endToEndStartTime': -1})


    # #Add Classifications
    # body.update({'detected_objects': frame.detected_objects})
    # body.update({'confidences': frame.confidences})

    return ('DATA' + '\r\n' + \
                'CSeq: ' + str(0) + '\r\n' + \
                'Content-Length: ' + str(getsizeof(body_str.encode())) + '\r\n' + \
                ujson.dumps(body) + '\r\n\r\n').encode()
    # return response.encode()

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

def generateResultMsg(frame):
    frame.stopServerProcTimer()
    # print('TESTING: ' + requestType.value)
    body = {}
    body_str = ""
    content_len = 0

    #Add metadata
    body.update({'mobileProcDeltaTime': frame.mobileProcDeltaTime})
    body.update({'serverProcDeltaTime': frame.serverProcDeltaTime})
    body.update({'transmitStartTime': frame.transmitStartTime})
    body.update({'endToEndStartTime': frame.endToEndStartTime})


    #Add Classifications
    body.update({'detected_objects': frame.detected_objects})
    body.update({'confidences': frame.confidences})

    return ('DATA' + '\r\n' + \
                'CSeq: ' + str(0) + '\r\n' + \
                'Content-Length: ' + str(getsizeof(body_str.encode())) + '\r\n' + \
                ujson.dumps(body) + '\r\n\r\n').encode()
    # return response.encode()

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
                global dataList
                global numPictures
                global numRECV
                global numSlices
                global inputShape
                global mobileProcTime 
                global transmitStartTime
                global endToEndStartTime
                global start


                data = self.request[0].strip()
                socket = self.request[1]
                msgParts = ujson.loads(data)
                reqType = RequestType[msgParts['MessageType']]

                
                if reqType == RequestType.HELLO:
                    print("HELLO Recv'd")
                    # Add output channel for response
                    resp = generateResponse(reqType)
                    print(self.client_address[0])
                    socket.sendto(resp, (self.client_address[0],controlPort))

                elif reqType == RequestType.SETUP:
                    print(msgParts)
                    print("SETUP Recv'd")
                    resp = generateResponse(reqType)
                    socket.sendto(resp, (self.client_address[0],controlPort))
                    isClientReady = True
                    numPictures = bytearray(msgParts['Payload'])


                elif reqType == RequestType.DATA_HEADER:
                    print("DATA_HEADER Recv'd")
                    numRECV = 0
                    numSlices = msgParts['NumSlices']
                    dataList = [None] * numSlices
                    mobileProcTime = msgParts['MobileDelta']
                    transmitStartTime = msgParts['TransmitStart']
                    endToEndStartTime = msgParts['TotalStart']
                    start = time.time()

                elif reqType == RequestType.DATA:
                    dataList[msgParts["Index"]] = msgParts["Payload"]
                    numRECV = numRECV + 1
                    if (None not in dataList):
                        result = preProcessData(dataList,self.inputShape,start)
                        input_queue.put(result)
                        results = generateResultMsg(self.output_queue.get())
                        socket.sendto(results, (self.client_address[0], controlPort))
                    else:
                        print(dataList)
                    

                elif reqType == RequestType.RESULTS:
                    output_dir = "distribGraphs"
                    print("Results recv'd")
                    

                    # with open('/'.join([output_dir, "raw_data.txt"]), "a") as f:
                    #     line = ",\t".join([object_detector.PARTITION_NAME, str(mobileTimings), str(mobileTimings), str(np.mean(transmitTimings)), str(np.var(transmitTimings)), str(np.mean(serverTimings)), str(np.var(serverTimings)), str(np.mean(end2EndTimings)), str(np.var(end2EndTimings))])
                    #     f.write(line)
                    #     f.write("\n")
                    

                elif reqType == RequestType.GOODBYE:
                    print(">>disconnect request recv'd")
                    results = generateResponse(RequestType.GOODBYE)
                    socket.sendto(results, (self.client_address[0], controlPort))
                else:
                    print("uhoh")

            def finish(self):
                pass

        ThreadingUDPServer.__init__(self, addr, MyUDPRequestHandler, bind_and_activate)
        ControlMixin.__init__(self, handler, poll_interval) 

    def setPartitionPt(self, partitionName, partitionDict):
        if partitionName in partitionDict.keys():
            self.inputShape = tuple(partitionDict[partitionName])
            print("Partition Point set to layer " + partitionName + " with shape " + str(self.inputShape))
        else:
            print("oops")

def process(dataList, someInputShape,start):
    with futures.ProcessPoolExecutor(2) as e: 
        futs = []
        futs.append(e.submit(preProcessData,dataList,someInputShape,start))
        res = [fut.result() for fut in futs]
        return res[0]

def preProcessData(dataList,someInputShape,start):
    if (object_detector.PARTITION_NAME == "Placeholder"):
        try:
            # print("interpreting as image")
            jpegBytes = bytearray(flattenDecode(dataList))
            start = time.time() * 1000
            img2 = cv2.imdecode(np.frombuffer(jpegBytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            img2 = img2.reshape((1, 227, 227, 3))
            frame = Frame(img2,mobileProcTime,transmitStartTime,endToEndStartTime,start)
            return frame
        except TypeError:
            print("TypeError Raised")

        # print("put")
        # results = generateResultMsg(self.output_queue.get())
        # socket.sendto(results, (self.client_address[0], controlPort))
            
    else:
        if DTYPE == "GSON":
            tfbytes = b64.decodebytes(bytearray(flatten(dataList)))
        else:
            # print("This is JACKSON")
            try:
                tfbytes = bytearray(flattenDecode(dataList))
                tfarr = np.frombuffer(tfbytes, dtype=np.float32)
                tfarr = tfarr.reshape(someInputShape)
                frame = Frame(tfarr,mobileProcTime,transmitStartTime,endToEndStartTime,start)
                return frame
            
            except TypeError:
                print("TypeError Raised")

            # self.input_queue.put(frame)
            # results = generateResultMsg(self.output_queue.get())
            # socket.sendto(results, (self.client_address[0], controlPort))


def main():
    input_q = Queue(maxsize=1)
    output_q = Queue(maxsize=1)
    object_detector.readPartitionData()

    #fig, ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True)

    udpserver = EasyUDPServer(input_q, output_q, ServerAddress, 0.01)
    
    udpserver.start()
    udpserver.setPartitionPt(object_detector.PARTITION_NAME, object_detector.partitions_dict)
    
    pool = Pool(1, object_detector.worker, (input_q, output_q))
    udpserver.serve_forever()
if __name__ == '__main__':   
    main()
