from sys import getsizeof
from enum import Enum
import ujson

class RequestType(Enum):
    HELLO = 'OK'
    SETUP = 'SETUP'
    READY_ACK = 'READY_ACK'
    PLAY = 'ACK'
    DATA_HEADER = 'DATA_HEADER_RECV'
    DATA = 'DATA_RECV'
    RESULTS = 6
    GOODBYE = 'GOODBYE'

def generateResponse(requestType, partitionName):
        # print('TESTING: ' + requestType.value)
        body = {}
        body_str = ""
        content_len = 0

        if (requestType == RequestType.SETUP):  #TODO: Add partition logic
            
            body.update({'Partition-Point': partitionName})
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

def generateErrorMsg():
    # frame.stopServerProcTimer()
    # print('TESTING: ' + requestType.value)
    body = {}
    body_str = ""
    content_len = 0

    #Add metadata
    body.update({'mobileProcDeltaTime': -1})
    body.update({'serverProcDeltaTime': -1})
    body.update({'transmitStartTime': -1})
    body.update({'endToEndStartTime': -1})


    # #Add Classifications
    # body.update({'detected_objects': frame.detected_objects})
    # body.update({'confidences': frame.confidences})

    return ('DATA' + '\r\n' + \
                'CSeq: ' + str(0) + '\r\n' + \
                'Content-Length: ' + str(getsizeof(body_str.encode())) + '\r\n' + \
                ujson.dumps(body) + '\r\n\r\n').encode()
    # return response.encode()

def generateResultMsg(frame):
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

    return ('DATA' + '\r\n' + \
                'CSeq: ' + str(0) + '\r\n' + \
                'Content-Length: ' + str(getsizeof(body_str.encode())) + '\r\n' + \
                ujson.dumps(body) + '\r\n\r\n').encode()
    # return response.encode()

