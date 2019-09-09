# voltage sampling
# Noah Johnson 5/1/19

import paramiko
import time
import select


hostname = "192.168.0.162"
port = 8022
user = 'noah'

root_command = "whoami\n"
root_command_result = "root"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname,port=port,username=user,password='')

transport = client.get_transport()
transport.connect(username=user,password='')
shell = client.invoke_shell()

while True:
    shell.send("su -c cat /sys/class/power_supply/battery/voltage_now")
    # Print data when available
    if shell != None and shell.recv_ready():
        alldata = shell.recv(1024)
        while shell.recv_ready():
            alldata += shell.recv(1024)
        strdata = str(alldata, "utf8")
        strdata.replace('\r', '')
        print(strdata, end = "")
        if(strdata.endswith("$ ")):
            print("\n$ ", end = "")
