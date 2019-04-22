import time
from multiprocessing import Queue

import base64 as b64
import cv2
import numpy as np


from object_detector import PARTITION_NAME
from utils.network_utils import RequestType
from utils.classification_utils import Frame


def flatten(l): return [item for sublist in l for item in sublist]


def flattenDecode(l): return [n for sublist in l for n in b64.b64decode(sublist)]






class Preprocessor():

    def __init__(self, input_q, output_q, inputShape):
        self.in_q = input_q
        self.out_q = output_q
        self.picList = []
        self.numPictures = 0
        self.numSlices = 0
        self.inputShape = inputShape

    def run(self):   

        # get input, which is a dictionary of msgParts
        while True:
            msgParts = self.in_q.get()
            reqType = RequestType[msgParts['MessageType']]

            #if the message is a setup msg then remember the number of pictures
            if reqType == RequestType.SETUP:
                self.numPictures = int(bytearray(msgParts['Payload']))
            # if the message is a dataheader, then...
            if reqType == RequestType.DATA_HEADER:
                self.numSlices = msgParts['NumSlices']
                self.picList = [None] * self.numSlices
                mobileProcTime = msgParts['MobileDelta']
                transmitStartTime = msgParts['TransmitStart']
                endToEndStartTime = msgParts['TotalStart']

            # if the message is a data fragment...
            if reqType == RequestType.DATA:
                # print("DATA Recv'd")
                # print(self.picList)

                try:
                    self.picList[msgParts["SliceIdx"]] = msgParts["Payload"]
                except IndexError:
                    self.picList = [None] * max(self.numSlices, 1)

                else:
                    if not (None in self.picList):
                        start = time.time()
                        print("done")
                        self.preProcessData(self.picList, self.inputShape, mobileProcTime, transmitStartTime, endToEndStartTime, start)
                        self.picList = [None]
                        print("reset PicList")


    def preProcessData(self, dataList, someInputShape, mobileProcTime, transmitStartTime, endToEndStartTime, start):
        if (PARTITION_NAME == "Placeholder"):
            try:
                print("interpreting as image")
                print(len(flattenDecode(dataList)))
                imgArr = np.array(flattenDecode(dataList), dtype=np.uint8)
                imgArr = imgArr.reshape((1, 227, 227, 3))
                frame = Frame(imgArr,mobileProcTime,transmitStartTime,endToEndStartTime,start)
                self.out_q.put(frame)
            except TypeError:
                print("TypeError Raised")

        else:
            try:
                tfbytes = bytearray(flattenDecode(dataList))

                tfarr = np.frombuffer(tfbytes, dtype=np.float32)
                tfarr = tfarr.reshape(someInputShape)
                frame = Frame(tfarr, mobileProcTime,
                            transmitStartTime, endToEndStartTime, start)
                self.out_q.put(frame)
            except TypeError:
                print("TypeError Raised")
