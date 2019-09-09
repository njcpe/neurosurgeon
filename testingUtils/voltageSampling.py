# voltage sampling
# Noah Johnson 5/1/19

import paramiko
import time
hostname = "192.168.0.162"
port = 8022
user = 'noah'

root_command = "whoami\n"
root_command_result = b"root"

def send_string_and_wait(command, wait_time, should_print):
    # Send the su command
    shell.send(command)

    # Wait a bit, if necessary
    time.sleep(wait_time)

    # Flush the receive buffer
    receive_buffer = shell.recv(1024)

    # Print the receive buffer, if necessary
    if should_print:
        print (receive_buffer)

def send_string_and_wait_for_string(command, wait_string, should_print):
    # Send the su command
    shell.send(command)

    # Create a new receive buffer
    receive_buffer = b''

    while not wait_string in receive_buffer:
        # Flush the receive buffer
        receive_buffer += shell.recv(1024)

    # Print the receive buffer, if necessary
    if should_print:
        print (receive_buffer)

#password auth is not supported with termux, so we must use RSA verification. this script assumes that the RSA verification is already set up;

client = paramiko.SSHClient()

client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect(hostname,port=port,username=user,password='')

shell = client.invoke_shell()


# Send the su command
send_string_and_wait("su\n", 1, True)

# Send the install command followed by a newline and wait for the done string
send_string_and_wait_for_string(root_command, root_command_result, True)

send_string_and_wait("cat /sys/class/power_supply/battery/voltage_now",1,True)
