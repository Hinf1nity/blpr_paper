import asyncio
import websockets
import socketserver
import socket

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

async def echo(websocket, path):
    while True:
        # Get user input
        user_input = input("Enter a message to send (type 'exit' to quit): ")
        
        # If the user types 'exit', close the connection
        if user_input.lower() == 'exit':
            print("Closing the server.")
            break
        
        # Send the user input to the connected client
        await websocket.send(user_input)
        print(f"Sent: {user_input}")

async def main():
    async with websockets.serve(echo, Ip_Dir, 8080):
        print("WebSocket server started at ws://"+Ip_Dir+":8080")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
