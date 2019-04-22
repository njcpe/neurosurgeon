import threading
import time
from multiprocessing import Pool, Process, Queue
from socketserver import DatagramRequestHandler, ThreadingUDPServer

import numpy as np

import object_detector
import ujson
from object_detector import PARTITION_NAME
from preprocessor import Preprocessor
from utils.network_utils import (RequestType, generateErrorMsg,
                                 generateResponse, generateResultMsg)


###
# Helper Functions for ensemble methods.
# Not currently used, but may be useful for other things, so retained
###

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

###
# End Helper Functions
###


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



def postProcess(socket, address, port, output_queue):
    '''Post Processor for handling returning the data to the client from the server.'''
    print("postProcessor Started")

    while True:
        #Get the result of the classification from the pipeline, this line will block until a new result is available.
        results = output_queue.get()
        #Stop the server processing timer.
        stopTime = time.time()

        #store this stop time in the results packet.
        results.stopServerProcTimer(stopTime)
        #generate the result message and send it to the client.
        socket.sendto(generateResultMsg(results),(address, port))
        print("sent")


# Subclass the DatagramRequestHandler
class ControlMixin(object):
    '''Defines the control methods for the threads within the UDP server.'''
    def __init__(self, handler, poll_interval):
        '''Executed when the threads are instantiated. Performs boilerplate setup.'''
        self._thread = None
        self.poll_interval = poll_interval
        self._handler = handler

    def start(self):
        '''Called when the start method is called. Instantiates a thread that will run the serve forever method of the server and daemonizes that thread.'''
        self._thread = t = threading.Thread(target=self.serve_forever,
                                            args=(self.poll_interval,))
        t.setDaemon(True)
        t.start()
        # print("server running")

    def stop(self):
        '''Boilerplate shutdown method'''
        self.shutdown()
        self._thread.join()
        self._thread = None



class EasyUDPServer(ControlMixin, ThreadingUDPServer):
    '''Class for handling UDP Requests to the server. When a Request is received, the handle() method of MyUDPRequestHandler is called.'''
    def __init__(self, input_queue, output_queue, addr, handler, poll_interval=0.001, bind_and_activate=True):
        '''Called when the class is instantiated. Sets up the processing pipeline and defines the class for handling UDP Requests.'''
        self.input_queue = input_queue
        self.output_queue = output_queue

        class MyUDPRequestHandler(DatagramRequestHandler):
            '''Class that defines how the server handles incoming requests.'''

            def __init__(self, request, client_address, server):
                '''Called when the handler is instantiated. Passes the processing pipeline from the server to the handler.'''
                self.input_queue = server.input_queue
                self.output_queue = server.output_queue
                self.inputShape = server.inputShape

                #calls the superclass initialize() method.
                DatagramRequestHandler.__init__(self, request, client_address, server)


            def handle(self):
                '''Method for actually handling an incoming request. Based on the request type, we either setup the session or pass data to the preprocessor for classification.'''
                # the data we want is the first element of the request, and we want to get rid of all the line endings (e.g., \r\n, \n)
                data = self.request[0].strip()
                # the socket object that the client sent from is the second element of the request.
                socket = self.request[1]
                # the message is received in the json format, so we unpack it into a set of key:value pairs, or a dictionary.
                msgParts = ujson.loads(data)
                #We match the request type of the message to the dictionary of possible request types.
                reqType = RequestType[msgParts['MessageType']]

                # then condition on the request type.
                if reqType == RequestType.HELLO:
                    #if the request is a "Hello" message, we need to let the client know we are here, so we echo the request
                    print("HELLO Recv'd")
                    #generate a response message from 
                    resp = generateResponse(reqType, PARTITION_NAME)
                    socket.sendto(resp, (self.client_address[0],controlPort))

                elif reqType == RequestType.SETUP:
                    print("SETUP Recv'd")
                    input_queue.put(msgParts)
                    resp = generateResponse(reqType, PARTITION_NAME)
                    socket.sendto(resp, (self.client_address[0], controlPort)) #Respond to setup request
                    postProcessor = Process(target=postProcess, args = (socket,self.client_address[0], controlPort,output_queue)) 
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
                    print("unknown request type!")

        ThreadingUDPServer.__init__(self, addr, MyUDPRequestHandler, bind_and_activate)
        ControlMixin.__init__(self, handler, poll_interval) 

    def setPartitionPt(self, partitionName, partitionDict):
        if partitionName in partitionDict.keys():
            self.inputShape = tuple(partitionDict[partitionName])
            print("Partition Point set to layer " + partitionName + " with shape " + str(self.inputShape))
        else:
            print("oops")
            print(partitionDict)

def main():
    #Setup the queues for the process communication between the Network Handler, Preprocessor, and Classifier
    preprocessor_input_q = Queue()
    preprocessor_output_q = Queue()
    classifier_output_q = Queue()

    object_detector.readPartitionData()


    udpserver = EasyUDPServer(preprocessor_input_q, classifier_output_q, ServerAddress, 0.01)
    udpserver.setPartitionPt(PARTITION_NAME, object_detector.partitions_dict)

    udpserver.start()



    try:

        preprocessor = Preprocessor(preprocessor_input_q,preprocessor_output_q, udpserver.inputShape)
        preprocessor_pool = Pool(1,preprocessor.run)
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
