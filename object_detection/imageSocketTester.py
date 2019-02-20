import sys
import msgpack
import socket
from enum import Enum
from io import BytesIO as bio

dataPorts = [4008, 4010]
nil = b"\xc0"


class RequestType(Enum):
    HELLO = 'OK'
    SETUP = 'SETUP'
    READY_ACK = 'READY_ACK'
    PLAY = 'ACK'
    DATA = 'DATA_RECV'
    PAUSE = 5
    STOP = 6
    TEARDOWN = 'GOODBYE'

def recvMsgpack(client):
    """receive one msgpack formatted packet from client. If socket is closed,
    empty string is returned even if some data received.

    IMPORTANT: The last part of the packet MUST be a .packNil()
    """
    bytebuf = []
    unpacker = msgpack.Unpacker(use_list=True, raw=False)
    
    while nil not in bytebuf:
        bytebuf = client.recv(1)
        # print(bytebuf)
        unpacker.feed(bytebuf)
    for m in unpacker:
        print(m)
        return m

def sendMsgpack(client, message):
    """
    Send a message as a msgpack encoded object (binary type)
    Takes: 
        client socket to send message (type socket)
        message dict of objects to send (type dict)
    """

    bytesout = bio()
    msgpack.pack(message, bytesout)
    msgpack.pack(None,bytesout) 
    bytesout.seek(0)
    for l in bytesout:
        print(l)
    bytesout.seek(0)
    client.send(bytesout.getvalue())



def generateResponse(requestType, CSeq):
    # print('TESTING: ' + requestType.value)
    response = {}
    
    response.update({'Request-Type': requestType})
    response.update({'CSeq': CSeq})

    
    if (requestType == RequestType.SETUP):  #TODO: Add partition logic

        response.update({'Partition-Point': 1})
        response.update({'UDP-Ports': dataPorts})

    return response

if __name__ == '__main__':
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.settimeout(1)
    server.bind(('',4002))
    server.listen(1)
    msg = []
    i = 0
    try:
        while True:
            client, address = server.accept()
            # client.setblocking(0)
            while True:
                msgDict = recvMsgpack(client)
                # print(msgDict)
                msg_type = msgDict.get('Request-Type','')
                msg_cseq = msgDict.get('CSeq', '')
                msg_data = msgDict.get('Body','')
                response = generateResponse(requestType=msg_type, CSeq=msg_cseq)  # add msg response handling, then test array send/recv dT
                sendMsgpack(client, response)

    except KeyboardInterrupt:
        server.close()
        sys.exit()
