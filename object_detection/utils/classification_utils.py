import msgpack
import ujson
import time
import numpy as np
class Frame:
    imageData = []
    input_shape = tuple()
    endToEndStartTime = 0
    endToEndDeltaTime = 0
    serverProcStartTime = 0
    serverProcDeltaTime = 0

    detected_objects = []
    confidences = []

    def __init__(self, json_def, input_shape):
        # print(msgpack.unpackb(json_def))
        self.__dict__ = ujson.loads(json_def)
        print(type(self.imageData[0]))
        self.input_shape = input_shape
        self.serverProcStartTime = time.time()
    
    def deleteRawImgData(self):
        self.imageData.clear()

    def getImageData(self):
        return np.reshape(np.concatenate([self.imageData]), newshape=self.input_shape)

    def stopServerProcTimer(self):
        self.serverProcDeltaTime = (time.time() - self.serverProcStartTime)*1000
        print("server proc timer stopped, tServer = " + str(self.serverProcDeltaTime) + " ms")
        print(type(self.serverProcDeltaTime))
