import socket
import os
import time
from utils.network_utils import NSCPServer

server = NSCPServer('', 20004)

try:
    while True:
        server.appendToMessageBuff(b'hey')
        time.sleep(5)
except KeyboardInterrupt:
    server.tcp_sock.close()
