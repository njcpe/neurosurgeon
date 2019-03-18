
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
    mobileProcDeltaTime = 0
    transmitStartTime = 0
    detected_objects = []
    confidences = []

    def __init__(self, imageData, mobileProcDeltaTime, transmitStartTime, endToEndStartTime):
        self.imageData = imageData
        self.mobileProcDeltaTime = mobileProcDeltaTime
        self.transmitStartTime = transmitStartTime
        self.serverProcStartTime = time.time()
        self.endToEndStartTime = endToEndStartTime
    
    def deleteRawImgData(self):
        self.imageData = None

    def getImageData(self):
        return self.imageData

    def stopServerProcTimer(self):
        self.serverProcDeltaTime = (time.time() - self.serverProcStartTime)*1000
        print("server proc timer stopped, tServer = " + str(self.serverProcDeltaTime) + " ms")
