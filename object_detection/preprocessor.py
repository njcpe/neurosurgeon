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


def preProcessData(dataList, someInputShape, mobileProcTime, transmitStartTime, endToEndStartTime, start):
    if (PARTITION_NAME == "Placeholder"):
        try:
            # print("interpreting as image")
            jpegBytes = bytearray(flattenDecode(dataList))            
            img2 = cv2.imdecode(np.frombuffer(jpegBytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            # cv2.imshow("test",img2)
            # cv2.waitKey(0)
            img2 = img2.reshape((1, 227, 227, 3))
            frame = Frame(img2,mobileProcTime,transmitStartTime,endToEndStartTime,start)
            return frame
        except TypeError:
            print("TypeError Raised")

    else:
        try:
            tfbytes = bytearray(flattenDecode(dataList))
            tfarr = np.frombuffer(tfbytes, dtype=np.float32)
            tfarr = tfarr.reshape(someInputShape)
            frame = Frame(tfarr, mobileProcTime,
                          transmitStartTime, endToEndStartTime, start)
            return frame

        except TypeError:
            print("TypeError Raised")


def PreprocessorWorker(input_q, output_q, inputShape):
    picList = []
    dataList = []
    numPictures = 0
    numSlices = 0

    # get input, which is a dictionary of msgParts
    while True:
        msgParts = input_q.get()
        reqType = RequestType[msgParts['MessageType']]

        #if the message is a setup msg then remember the number of pictures
        if reqType == RequestType.SETUP:
            numPictures = int(bytearray(msgParts['Payload']))
            picList = [None] * numPictures
        # if the message is a dataheader, then...
        if reqType == RequestType.DATA_HEADER:
            # print("DATA_HEADER Recv'd")
            # print(msgParts["PicIdx"])
            # print(msgParts["NumSlices"])
            picIdx = msgParts["PicIdx"]
            numSlices = msgParts['NumSlices']
            picList[picIdx] = [None] * numSlices
            # print(picList)
            mobileProcTime = msgParts['MobileDelta']
            transmitStartTime = msgParts['TransmitStart']
            endToEndStartTime = msgParts['TotalStart']

        # if the message is a data fragment...
        if reqType == RequestType.DATA:
            print("DATA Recv'd")

            # print(msgParts["PicIdx"])
            # print(msgParts["SliceIdx"])


            picList[msgParts["PicIdx"]][msgParts["SliceIdx"]] = msgParts["Payload"]
            # print(picList)

            if not any(None in sublist for sublist in picList):
                start = time.time()
                print("done")
                for pic in picList:
                    output_q.put(preProcessData(pic, inputShape, mobileProcTime, transmitStartTime, endToEndStartTime, start))
                picList = [None] * numPictures
                print("reset PicList")
            # elif(msgParts["PicIdx"] == numPictures-1 ):
            #     coordsList = [[x, y] for x, li in enumerate(picList) for y, val in enumerate(li) if val == None]
            #     print(coordsList)
            # else:
                # print(dataList)
        # set to correct place

        # increment numRecv

        # if the list is complete...
        # preProcess the list into frame
        # put frame into output_q
        # else...
        # pass?
