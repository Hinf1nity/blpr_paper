import asyncio
import sys
import http.server
import socketserver
import socket
from websockets.server import serve
from threading import Thread

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception as e:
        print(f"Error: {e}")
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

Ip_Dir = get_ip_address()
HTTP_PORT = 8000
WEBSOCKET_PORT = 8080
clients = []

print("Server Ip Address : ", Ip_Dir)


class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.ip_dir = Ip_Dir
        self.websocket_port = WEBSOCKET_PORT
        super().__init__(*args, directory="./webpage/", **kwargs)

    def end_headers(self):
        self.send_header("X-Ip-Dir", self.ip_dir)
        self.send_header("X-WebSocket-Port", str(self.websocket_port))
        super().end_headers()

    def log_message(self, format, *args):
        pass


def start_file_server():
    with socketserver.TCPServer(("", HTTP_PORT), SimpleHTTPRequestHandler) as httpd:
        print("File server started at port", HTTP_PORT)
        httpd.serve_forever()


async def echo(websocket):
    clients.append(websocket)
    async for message in websocket:
        print("Received message:", message)
        # if(message=="Give_Clients"):
        # await websocket.send("Number of clients : " + str(len(clients)))
        for client in clients:
            await client.send(message)


async def main():
    async with serve(echo, Ip_Dir, WEBSOCKET_PORT):
        print("Websocket server started at port", WEBSOCKET_PORT)
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    t = Thread(target=start_file_server, daemon=True).start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)