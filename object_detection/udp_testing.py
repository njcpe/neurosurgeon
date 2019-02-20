# Sample UDP Server - Multi threaded
raw_input = input

from utils.classification_utils import Frame
from socketserver import ThreadingUDPServer, DatagramRequestHandler
import threading
import ujson
from enum import Enum
import csv

from sys import getsizeof
import numpy as np
import base64 as b64
import io
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool, Queue
import sys
from imageio import imread
import time
import object_detector

import queue
class RequestType(Enum):
    HELLO = 'OK'
    SETUP = 'SETUP'
    READY_ACK = 'READY_ACK'
    PLAY = 'ACK'
    DATA_HEADER = 'DATA_HEADER_RECV'
    DATA = 'DATA_RECV'
    LAST_DATA = 'LAST_DATA_RECV'
    PAUSE = 5
    STOP = 6
    TEARDOWN = 'GOODBYE'

def generateResponse(requestType):
        # print('TESTING: ' + requestType.value)
        body = {}
        body_str = ""
        content_len = 0

        if (requestType == RequestType.SETUP):  #TODO: Add partition logic
            
            body.update({'Partition-Point': 1})
            body.update({'UDP-Ports': [9998,9999]})

            body_str = ujson.dumps(body)
            content_len = getsizeof(body_str.encode())

        response = requestType.value + '\r\n' + \
                    'CSeq: ' + str(0) + '\r\n' + \
                    'Content-Length: ' + str(content_len) + '\r\n' + \
                    body_str + '\r\n\r\n'
        return response.encode()

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


    body_str = ujson.dumps(body)
    content_len = getsizeof(body_str.encode())

    response = 'DATA' + '\r\n' + \
                'CSeq: ' + str(0) + '\r\n' + \
                'Content-Length: ' + str(content_len) + '\r\n' + \
                body_str + '\r\n\r\n'
    return response.encode()

controlPort = 9998 

dataList = []
numRECV = 0
numSlices = -1
inputShape = (1, 9216)
mobileProcTime = 0
transmitStartTime = 0
endToEndStartTime = 0


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
    def __init__(self, input_queue, output_queue, addr, handler, poll_interval=0.5, bind_and_activate=True):
        self.input_queue = input_queue
        self.output_queue = output_queue
        # print("server init called")

    

        
        class MyUDPRequestHandler(DatagramRequestHandler):

            def __init__(self, request, client_address, server):
                self.input_queue = server.input_queue
                self.output_queue = server.output_queue
                DatagramRequestHandler.__init__(self, request, client_address, server)




            def handle(self):
                global dataList
                global numRECV
                global numSlices
                global inputShape
                global mobileProcTime 
                global transmitStartTime
                global endToEndStartTime

                flatten = lambda l: [item for sublist in l for item in sublist]
                data = self.request[0].strip()
                socket = self.request[1]
                msgParts = ujson.loads(data)

                reqType = RequestType[msgParts['Message-Type']]

                if reqType == RequestType.HELLO:
                    print("HELLO Recv'd")
                    # Add output channel for response
                    resp = generateResponse(reqType)
                    print(resp)
                    socket.sendto(resp, (self.client_address[0],controlPort))


                elif reqType == RequestType.SETUP:
                    print("SETUP Recv'd")
                    
                    resp = generateResponse(reqType)
                    socket.sendto(resp, (self.client_address[0],controlPort))
                    isClientReady = True

                elif reqType == RequestType.DATA_HEADER:
                    numRECV = 0
                    print("DATA_HEADER Recv'd")
                    numSlices = msgParts['Num-Slices']
                    dataList = [None] * numSlices
                    mobileProcTime = msgParts['Mobile-Delta']
                    transmitStartTime = msgParts['Transmit-Start']
                    endToEndStartTime = msgParts['Total-Start']

                elif reqType == RequestType.DATA:
                    numRECV = numRECV + 1;
                    if numRECV < numSlices:
                        # print(numRECV)
                        dataList[msgParts["Index"]] = msgParts["Payload"]
                    else:
                        dataList[msgParts["Index"]] = msgParts["Payload"]
                        start = time.time() * 1000

                        dataList = flatten(dataList) #Flatten all lists into one
                        jpegStrB64 = bytearray(dataList).decode("UTF-8")  #convert the list of bytes as int values to a byte array, decode that byte array into a string (BASE64)
                        img2 = cv2.imdecode(np.fromstring(b64.b64decode(jpegStrB64), np.uint8), cv2.IMREAD_COLOR)
                        img2 = img2.reshape((1, 227, 227, 3))
                        print((time.time()*1000) - start)
                        frame = Frame(img2,mobileProcTime,transmitStartTime,endToEndStartTime)
                        self.input_queue.put(frame)
                        print("put")
                        results = generateResultMsg(self.output_queue.get())
                        socket.sendto(results, (self.client_address[0],controlPort))

                elif reqType == RequestType.TEARDOWN:
                    print(">>disconnect request recv'd")
                    # self.isClientReady = False
                    # self.isClientConnected = False
                    # print("Removing client at " + str(s.getpeername()
                    #                                 [0]) + ':' + str(s.getpeername()[1]))
                    # if s in self.inputs:
                    #     self.inputs.remove(s)
                    #     print("removed from inputs")
                    # if s in self.outputs:
                    #     self.outputs.remove(s)
                    #     print("removed from outputs")
                    
                    # for k in self.message_queues.keys():
                    #     print(k)
                    #     print(type(k))
                    
                    # del self.message_queues[s.getpeername()[0]]
                else:
                    print("uhoh")

        ThreadingUDPServer.__init__(self, addr, MyUDPRequestHandler, bind_and_activate)
        ControlMixin.__init__(self, handler, poll_interval)

def main():
    input_q = Queue(maxsize=1)
    output_q = Queue(maxsize=1)
    
    udpserver = EasyUDPServer(input_q, output_q, ServerAddress, 0.0001)
    udpserver.start()
    pool = Pool(1, object_detector.worker, (input_q, output_q))
    udpserver.serve_forever()
if __name__ == '__main__':
    main()
    

# Create a Server Instance
# server = socketserver.ThreadingUDPServer(ServerAddress, MyUDPRequestHandler)
# Make the server wait forever serving connections
# server.serve_forever()

