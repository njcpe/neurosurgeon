
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

    def __init__(self, imageData, mobileProcDeltaTime, transmitStartTime, endToEndStartTime, syncStartTime):
        self.imageData = imageData
        self.mobileProcDeltaTime = mobileProcDeltaTime
        self.transmitStartTime = transmitStartTime
        self.serverProcStartTime = syncStartTime
        self.endToEndStartTime = endToEndStartTime
    
    def deleteRawImgData(self):
        self.imageData = None

    def getImageData(self):
        return self.imageData

    def stopServerProcTimer(self,syncEndTime):
        self.serverProcDeltaTime = (syncEndTime - self.serverProcStartTime)*1000*1000000 #convert server proc time from seconds to milliseconds to nanoseconds 
        print("server proc timer stopped, tServer = " + str(self.serverProcDeltaTime) + " ns")
