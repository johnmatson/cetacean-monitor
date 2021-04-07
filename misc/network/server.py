import socket

model = Model()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(),1234))
s.listen(5)

ts = model.predict(data).toString()

white True:
    clientsocket, address = s.accept()
    print(f"Connection from address: {address}")
    clientsocket.send(bytes(ts, "utf-8"))
    clietsocket.close()