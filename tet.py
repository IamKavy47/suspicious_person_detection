import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(("192.168.29.74", 4747))
if result == 0:
    print("Connection successful!")
else:
    print(f"Connection failed with error code {result}")
sock.close()
