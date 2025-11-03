import asyncio
import websockets

# List to store the received codes
received_codes = []

# List of fixed random 6-character codes
fixed_random_codes = [
    "A1B2C3", "D4E5F6", "G7H8I9"
    ]

# WebSocket listener to add/delete codes from the list and print every message
async def listen_websocket(uri, stop_event):
    while True:  # Retry connection indefinitely
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to WebSocket.")
                stop_event.set()  # Allow code printing when connected
                async for message in websocket:
                    print(f"Message received: {message}")  # Print every message received
                    parts = message.split('_')

                    # Check if the message is formatted correctly
                    if len(parts) == 2:  # Expecting two parts: command and code
                        command = parts[0]
                        code = parts[1]
                        if command == "add":
                            if code not in received_codes:
                                received_codes.append(code)
                                print(f"Code {code} added")
                        elif command == "delete":
                            if code in received_codes:
                                received_codes.remove(code)
                                print(f"Code {code} deleted")
                    else:
                        print(f"Invalid message format: {message}. Expected format: 'add_<code>' or 'delete_<code>'")
        except (websockets.ConnectionClosedError, websockets.InvalidURI, OSError) as e:
            print(f"WebSocket connection failed or closed: {e}. Retrying in 2 seconds...")
            stop_event.clear()  # Halt code printing when WebSocket is down
            await asyncio.sleep(2)  # Wait before retrying

# Task to print the fixed codes and check if they match with the received codes
async def print_fixed_codes(stop_event):
    try:
        while True:
            await stop_event.wait()  # Wait until WebSocket is connected
            for fixed_code in fixed_random_codes:
                print(f"Fixed Code: {fixed_code}")
                if fixed_code in received_codes:
                    print("Code found!")
                await asyncio.sleep(2)  # Wait 1 second between code prints
    except asyncio.CancelledError:
        print("Code printing task cancelled. Cleaning up...")

# Main function to run both tasks in parallel
async def main():
    websocket_uri = 'ws://192.168.1.91:8080'  # Replace with your WebSocket server URI
    stop_event = asyncio.Event()  # Event to control code printing

    # Create code printing task (it will pause based on stop_event)
    code_printing_task = asyncio.create_task(print_fixed_codes(stop_event))

    while True:
        # Create WebSocket connection task and ensure it controls the code printing task
        websocket_task = asyncio.create_task(listen_websocket(websocket_uri, stop_event))

        try:
            await websocket_task  # Keep retrying WebSocket connection
        except asyncio.CancelledError:
            print("Main task cancelled. Cleaning up...")
            stop_event.clear()  # Stop code printing when WebSocket is disconnected
            code_printing_task.cancel()
            try:
                await code_printing_task  # Wait for the code printing task to finish
            except asyncio.CancelledError:
                print("Code printing task cancelled during cleanup.")
            break

        # If WebSocket fails, halt code printing and retry WebSocket connection
        stop_event.clear()  # Stop code printing
        await websocket_task  # Retry WebSocket connection

# Entry point to run the async tasks with graceful shutdown
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Shutting down...")
