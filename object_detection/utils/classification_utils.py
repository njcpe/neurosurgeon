import ujson
import time

class Frame:
    imageData = []
    endToEndStartTime = 0
    endToEndDeltaTime = 0
    serverProcStartTime = 0
    serverProcDeltaTime = 0

    def __init__(self, json_def):
        self.__dict__ = ujson.loads(json_def)
    
    def startServerProcTimer(self):
        self.serverProcStartTime = time.time()
        print("server proc timer started")

    def stopServerProcTimer(self):
        self.serverProcDeltaTime = time.time() - self.serverProcStartTime
        print("server proc timer stopped")