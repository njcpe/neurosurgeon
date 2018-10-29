import queue
import re
import select
import socket
import threading
import time
import timeit
from enum import Enum
from sys import getsizeof

from utils.classification_utils import Frame

import numpy as np

import tensorflow as tf
import json  # TODO: migrate to msgpack!
import ujson  # TODO: migrate to msgpack!


class RequestType(Enum):
    HELLO = 'OK'
    SETUP = 'SETUP'
    READY_ACK = 'READY_ACK'
    PLAY = 'ACK'
    DATA = 'DATA_RECV'
    PAUSE = 5
    STOP = 6
    TEARDOWN = 'GOODBYE'

def brkpt():
    input("Press Any Key to continue")

def recv_exactly(sock, count): #FIXME: This is a messy solution, make it better, more stable if possible
    """receive exactly count bytes from socket. If socket is closed,
    empty string is returned even if some data received.
    """
    buf = []
    while count:
        try:
            data = sock.recv(count)
        except BlockingIOError:
            continue
        count -= len(data)
        buf.append(data)
        # print(buf)
    return b''.join(buf)

class NSCPServer(object):
    """ Neurosurgeon Control Protocol Server
    Handles all control protocols for Neurosurgeon
    The run() method will be started and it will run in the background
    until the application exits.
    """


    def __init__(self, host, port, interval=1):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        #TCP Setup

        self.HEADER_SIZE = 60


        self.interval = interval
        self.host = host
        self.port = port
        self.isClientReady = False
        self.isClientConnected = False
        self.sessControlSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sessControlSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sessControlSock.bind((self.host, self.port))
        self.message_queues = {}

        self.oldData = ''

        self.movingAvgN = 5

        #TCP Setup (DATA)
        self.dataSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dataSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.dataPorts = [4008, 4010]  #UDP Port array defaults, [0] is the port that server recieves on, [1] is the one that the server will target.
        self.partitionDataQueue = queue.Queue(maxsize=2)
        self.dataSock.bind(('', self.dataPorts[0]))
        self.dataCSeq = 0

        self.dataList = [] #empty data list to be used later for concat'ing sections of array.

        self.inputs = [self.sessControlSock, self.dataSock]
        self.outputs = []

        self.tServerList = []
        self.avgTServer = 0

        self.inputShape = (1,9216)

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                 # Start the execution

    def read(self):
        '''
        Returns the next tuple of the form (float array, socket object) from the dataQueue.
        Throws Empty Exception.
        '''
        dataBundle = self.partitionDataQueue.get_nowait()
        # print(dataBundle)
        return dataBundle

    def generateResponse(self, requestType, CSeq):
        # print('TESTING: ' + requestType.value)
        body = {}
        body_str = ""
        content_len = 0

        if (requestType == RequestType.SETUP):  #TODO: Add partition logic
            
            body.update({'Partition-Point': 1})
            body.update({'UDP-Ports': self.dataPorts})

            body_str = ujson.dumps(body)
            content_len = getsizeof(body_str.encode())

        response = requestType.value + '\r\n' + \
                    'CSeq: ' + str(CSeq) + '\r\n' + \
                    'Content-Length: ' + str(content_len) + '\r\n' + \
                    body_str + '\r\n\r\n'


        # print("RESPONSE: " + response)
        return response.encode()
    
    def setPartitionPt(self, partitionName, partitionDict):
        if partitionName in partitionDict.keys():
            self.inputShape = tuple(partitionDict[partitionName])
            print("Partition Point set to layer " + partitionName + " with shape " + str(self.inputShape))


    def run(self):
        self.sessControlSock.listen(5)
        self.dataSock.listen(5)
        # print('>>Listening')
        while self.inputs:
            # print(self.inputs[0].getsockname())
            readable, writeable, errored = select.select(
                self.inputs, self.outputs, self.inputs)

            for s in readable:
                if s is self.sessControlSock:
                    client, address = self.sessControlSock.accept()
                    self.inputs.append(client)
                    print("connection from:", address)
                    client.setblocking(0)
                    self.message_queues[client] = queue.Queue(maxsize=2)

                elif s is self.dataSock:
                    client, address = self.dataSock.accept()
                    self.inputs.append(client)
                    print("data connection from: ", address)
                    client.setblocking(0)
                    self.message_queues[client] = queue.Queue(maxsize=2)

                else:
                    # Recv and parse Header
                    head = recv_exactly(s, self.HEADER_SIZE)
                    head_str = (head.decode('utf-8')).strip()
                    print(head_str)
                    if head_str:
                        headerParts = (head_str).splitlines()
                        try:
                            reqType = RequestType[headerParts[0]]
                            cSeq = int((str(headerParts[1]).replace(" ", "")).partition(':')[2])
                            bodyLength = int((str(headerParts[2]).replace(" ", "")).partition(':')[2])
                            lastPartFlag = int((str(headerParts[3]).replace(" ", '')).partition(':')[2])
                        except:
                            print('Index Error')
                            exit

                        # print('>>Request Received:\n' + head_str)
                        body = recv_exactly(s, bodyLength)
                        if reqType == RequestType.HELLO:
                            print("BODY IS: " + str(body))
                            body = ujson.loads(bytes.decode(body))
                            self.isClientConnected = True
                            if s not in self.outputs:
                                self.outputs.append(s)
                            # Add output channel for response
                            resp = self.generateResponse(reqType, cSeq)
                            self.message_queues[s].put(resp)

                        elif reqType == RequestType.SETUP:
                            #Client can specify ports 
                            ports = re.search(rb'\d{4}', body)
                            if (ports != None and not np.array_equal(ports,self.dataPorts)):
                                for i in ports:
                                    self.dataPorts[i] = int(ports[i])
                                    self.dataSock.bind(('',self.dataPorts[0]))
                            # print("SETUP Data Transfer on port "+ str(self.dataPorts[0]))
                            resp = self.generateResponse(reqType, cSeq)
                            self.message_queues[s].put(resp)
                            self.isClientReady = True

                        elif reqType == RequestType.DATA:
                            '''
                            deserialize JSON float array into np.ndarray with appropriate shape.
                            '''
                            if s not in self.outputs:
                                self.outputs.append(s)
                            start = time.time()
                            try: 
                                # self.dataList.append(np.asarray(ujson.loads(body), dtype=float))
                                self.dataList.append(body)
                                # print(">>data Appended, current len is: " + str(len(self.dataList)))
                            except:
                                print('Decode Error')
                                break
                            
                            if lastPartFlag:
                                # print('>>Last Part of Message Recv\'d')
                                dataArr = b''.join(self.dataList)
                                newestFrame = Frame(dataArr, self.inputShape)

                                # print(newestFrame.__dict__(2))
                                self.partitionDataQueue.put(newestFrame)
                                self.dataList.clear()

                        elif reqType == RequestType.TEARDOWN:
                            print(">>disconnect request recv'd")
                            self.isClientReady = False
                            self.isClientConnected = False
                            print("Removing client at " + str(s.getpeername()
                                                            [0]) + ':' + str(s.getpeername()[1]))
                            if s in self.inputs:
                                self.inputs.remove(s)
                                print("removed from inputs")
                            if s in self.outputs:
                                self.outputs.remove(s)
                                print("removed from outputs")
                            
                            for k in self.message_queues.keys():
                                print(k)
                                print(type(k))
                            
                            del self.message_queues[s.getpeername()[0]]


                        else:
                            print("you shouldn't be here")
                            self.message_queues[s].put(self.generateResponse(reqType, cSeq))
                    else:
                        pass

            for s in writeable:
                try:
                    next_msg = self.message_queues[s].get_nowait()
                except queue.Empty:
                    # print('queue is empty')
                    pass
                except KeyError:
                    print('queue for ' + str(s.getpeername()) + ' does not exist!')
                    pass
                else:
                    totalsent = 0
                    sizeObj = len(next_msg)
                    while (totalsent < sizeObj and self.isClientConnected):
                        sent = s.send(next_msg[totalsent:])
                        print("Sent")
                        s.send(b'\n')
                        if sent == 0:
                            raise RuntimeError('Socket is broke')
                        totalsent += sent

            for s in errored:
                print('>>handling exceptional condition for :' + str(s.getpeername()))
                self.inputs.remove(s)
                if s in self.outputs:
                    self.outputs.remove(s)
                s.close()

                del self.message_queues[s]
        
    def generateDataMsg(self, body, dataCSeq):
        body.stopServerProcTimer()
        body_serialized = ujson.dumps(body)
        print(body_serialized)
        content_len = getsizeof(body_serialized.encode())
        seq = [
            'DATA', '\r\n',
            'CSeq: ', str(dataCSeq), '\r\n',
            'Content-Length: ', str(content_len), '\r\n',
            body_serialized, '\r\n\r\n'
        ]  
        message = ''.join(seq)
        return message.encode()

    def appendToMessageBuff(self, data):
        for s in self.outputs:
            if (self.message_queues[s].full() == False and s.getsockname()[1] == self.dataPorts[0]):
                self.dataCSeq += 1
                self.message_queues[s].put(self.generateDataMsg(data, self.dataCSeq))
            else:
                pass
            # print("appended to obuff for " + s.getpeername()[0])
        # else:
        #     print('msg queue for ' + str(src.getpeername()) + ' does not exist!')
            # for s in self.outputs:
            #     print('OUTPUT: ' + str(s))
            # for s in self.inputs:
            #     print('INPUT: ' + str(s))
            # for s in self.message_queues:
            #     print('MESSAGE QUEUE: ' + str(s))
            # print('SRC: ' + str(src))
