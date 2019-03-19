from __future__ import print_function

from twisted.internet.protocol import DatagramProtocol
from twisted.internet import reactor,threads
import ujson
from enum import Enum
from sys import getsizeof
import object_detector
import base64 as b64
import numpy as np
import cv2
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

controlPort = 9998 

class Echo(DatagramProtocol):
    
    dataList = None
    pictureList = None
    numRECV = 0
    numSlices = 0
    inputShape = None
    numPicturesRecv = 0
    numPictures = 0


    def datagramReceived(self, data, addr):
        threads.deferToThread(self.handle,data,addr)

    def handle(self, data, addr):
        
        # global mobileProcTime 
        # global transmitStartTime
        # global endToEndStartTime

        flatten = lambda l: [item for sublist in l for item in sublist]
        flattenDecode = lambda l: [n for sublist in l for n in b64.b64decode(sublist)]

        msgParts = ujson.loads(data)
        reqType = RequestType[msgParts['MessageType']]
        
        destAddr = (addr[0],controlPort)
        
        if reqType == RequestType.HELLO:
            print("HELLO Recv'd")
            # Add output channel for response
            resp = generateResponse(reqType)
            #print(resp)
            print(addr[0])
            self.transport.write(resp, destAddr)


        elif reqType == RequestType.SETUP:
            print(msgParts)
            print("SETUP Recv'd")
            resp = generateResponse(reqType)
            self.transport.write(resp, destAddr)
            self.numPictures = int(bytearray(msgParts['Payload']).decode('utf-8'))
            print(self.numPictures)
            self.numPicturesRecv = 0
            self.pictureList = [None]*self.numPictures

            print(len(self.pictureList))

        elif reqType == RequestType.DATA_HEADER:
            self.numRECV = 0
            self.numPicturesRecv = self.numPicturesRecv + 1
            print("DATA_HEADER Recv'd")
            self.numSlices = msgParts['NumSlices']
            self.dataList = [None] * self.numSlices
            self.mobileProcTime = msgParts['MobileDelta']
            self.transmitStartTime = msgParts['TransmitStart']
            self.endToEndStartTime = msgParts['TotalStart']
            print(msgParts)

        elif reqType == RequestType.DATA:
            self.numRECV = self.numRECV + 1;
            if self.numRECV < self.numSlices:
                # print("how many left: " + str(self.numSlices - self.numRECV))
                  
                self.dataList[msgParts["Index"]] = msgParts["Payload"]
            elif self.numPicturesRecv >= self.numPictures:
                self.dataList[msgParts["Index"]] = msgParts["Payload"]
                self.pictureList[self.numPicturesRecv - 1] = self.dataList
                # print(self.pictureList)
                # start = time.time() * 1000
                # print(msgParts)
                print("done recving")
                if (object_detector.PARTITION_NAME == "Placeholder"):

                    for image in self.pictureList:
                        print("interpreting as image")
                        jpegBytes = bytearray(flattenDecode(image))
                        img2 = cv2.imdecode(np.frombuffer(jpegBytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        img2 = img2.reshape((1, 227, 227, 3))
                        # print(img2)
                        # cv2.imshow("win", img2)
                        # cv2.waitKey(0)
                        # frame = Frame(img2,mobileProcTime,transmitStartTime,endToEndStartTime, start)
                        # self.input_queue.put(frame)
                        # results = generateResultMsg(self.output_queue.get())
                        # socket.sendto(results, (self.client_address[0], controlPort))
                        
                # else:
                #     for image in pictureList:
                #         print(image)    
                #     # for image in pictureList:
                #     #     tfbytes = bytearray(flattenDecode(image))
                #     #     tfarr = np.frombuffer(tfbytes, dtype=np.float32)
                #     #     tfarr = tfarr.reshape(self.inputShape)
                #     #     frame = Frame(tfarr,mobileProcTime,transmitStartTime,endToEndStartTime, start)
                #     #     self.input_queue.put(frame)
                #     #     results = generateResultMsg(self.output_queue.get())
                #     #     socket.sendto(results, (self.client_address[0], controlPort))
            else:
                self.dataList[msgParts["Index"]] = msgParts["Payload"]
                self.pictureList[self.numPicturesRecv - 1] = self.dataList

                print(self.pictureList[self.numPicturesRecv - 1])
            

        elif reqType == RequestType.GOODBYE:
            print(">>disconnect request recv'd")
            results = generateResponse(RequestType.GOODBYE)
            socket.sendto(results, (self.client_address[0], controlPort))
        else:
            print("uhoh")


reactor.listenUDP(9998, Echo())
reactor.run()
