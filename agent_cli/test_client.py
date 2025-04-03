# test_client.py

import asyncio
import socketio
import sys
import aioconsole # For async input
import requests # To perform login
import uuid
import logging
import argparse

# Configure basic logging for the client
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - CLIENT - %(levelname)s - %(message)s')

# Global state for the client
sio = socketio.AsyncClient(logger=True, engineio_logger=False) # Set engineio_logger=True for transport details
is_connected = False
auth_token = None
current_conversation_id = None

# Server URL (replace if your backend runs elsewhere)
DEFAULT_SERVER_URL = "http://localhost:5001"
DEFAULT_API_URL = f"{DEFAULT_SERVER_URL}/api/auth"


@sio.event
async def connect():
    global is_connected
    print("\n[+] Successfully connected to the server.")
    is_connected = True
    # Optionally, join a conversation room or send initial setup message
    print("[-] Type '/login <username> <password>' or '/register <username> <email> <password>' to authenticate.")
    print("[-] Type '/new' to start a new conversation (or get a new ID).")
    print("[-] Type your message to chat, or '/quit' to exit.")


@sio.event
async def connect_error(data):
    global is_connected
    print(f"\n[!] Connection failed: {data}")
    is_connected = False


@sio.event
async def disconnect():
    global is_connected
    print("\n[!] Disconnected from the server.")
    is_connected = False


@sio.event
async def connection_ack(data):
    """Handle connection acknowledgement from server."""
    print(f"[*] Server ACK: {data.get('message')} (SID: {data.get('sid')})")


@sio.event
async def partial_response(data):
    """Handle incoming partial chat messages from the server."""
    # Print response chunk without newline to simulate streaming
    print(f"\r>>> Assistant: {data.get('message')}", end='', flush=True)


@sio.event
async def response_complete(data):
    """Handle indication that a response stream is complete."""
    # Print a newline after the streaming is done
    print() # Moves cursor to next line after partial_response
    print(f"[*] Server: {data.get('message')}") # Optional completion message
    # Ready for next input prompt
    await print_prompt()


@sio.event
async def error(data):
    """Handle error messages from the server."""
    print(f"\n[!] Server Error: {data.get('message')}")
    await print_prompt() # Show prompt again after error


async def print_prompt():
    """Prints the input prompt."""
    conv_id_str = f" (Conv: {current_conversation_id})" if current_conversation_id else ""
    token_str = " (Authenticated)" if auth_token else ""
    prompt = f"\rYou{conv_id_str}{token_str}: "
    # Use aioconsole.aprint to avoid messing with current input line
    await aioconsole.aprint(prompt, end='')


async def login(username, password, api_url):
    """Logs in via REST API and stores the token."""
    global auth_token
    try:
        print(f"\n[*] Attempting login as {username}...")
        response = requests.post(f"{api_url}/login", json={
            "username": username,
            "password": password
        })
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        data = response.json()
        auth_token = data.get("access_token")
        if auth_token:
            print("[+] Login successful. Token stored.")
            # If connected, potentially disconnect and reconnect with token?
            # Or send an auth update message if the server supports it.
            # For simplicity, we'll use the token on the *next* connection.
            if is_connected:
                 print("[!] Token will be used on next connection.")
        else:
            print("[!] Login failed: No access token received.")
    except requests.exceptions.RequestException as e:
        print(f"[!] Login HTTP request failed: {e}")
        if e.response is not None:
             print(f"    Response: {e.response.status_code} - {e.response.text}")
    except Exception as e:
         print(f"[!] Login failed with unexpected error: {e}")


async def register(username, email, password, api_url):
    """Registers a new user via REST API."""
    try:
        print(f"\n[*] Attempting registration for {username}...")
        response = requests.post(f"{api_url}/register", json={
            "username": username,
            "email": email,
            "password": password
        })
        response.raise_for_status()
        data = response.json()
        print(f"[+] Registration successful: {data.get('message')}")
    except requests.exceptions.RequestException as e:
        print(f"[!] Registration HTTP request failed: {e}")
        if e.response is not None:
             print(f"    Response: {e.response.status_code} - {e.response.text}")
    except Exception as e:
         print(f"[!] Registration failed with unexpected error: {e}")


async def main_loop(server_url):
    global current_conversation_id
    global is_connected

    while True:
        try:
            if not is_connected:
                print(f"\n[*] Attempting to connect to {server_url}...")
                auth_payload = {'token': auth_token} if auth_token else {}
                await sio.connect(server_url, auth=auth_payload, transports=['websocket'])
                # Wait a bit for connection events to fire
                await asyncio.sleep(0.5)
                if not is_connected:
                     print("[!] Failed to connect. Retrying in 5 seconds...")
                     await asyncio.sleep(5)
                     continue # Retry connection

            # If connected, show prompt and wait for input
            await print_prompt()
            line = await aioconsole.ainput() # Use async input

            if not line:
                continue

            if line.lower() == '/quit':
                print("[*] Disconnecting...")
                await sio.disconnect()
                break

            elif line.lower().startswith('/login '):
                parts = line.split(' ', 3)
                if len(parts) == 3:
                    await login(parts[1], parts[2], DEFAULT_API_URL)
                else:
                    print("[!] Usage: /login <username> <password>")

            elif line.lower().startswith('/register '):
                 parts = line.split(' ', 4)
                 if len(parts) == 4:
                     await register(parts[1], parts[2], parts[3], DEFAULT_API_URL)
                 else:
                     print("[!] Usage: /register <username> <email> <password>")

            elif line.lower() == '/new':
                current_conversation_id = str(uuid.uuid4())
                print(f"[*] Started new conversation: {current_conversation_id}")

            elif line.lower() == '/help':
                 print("\nCommands:")
                 print("  /login <user> <pass> - Log in via API")
                 print("  /register <user> <email> <pass> - Register via API")
                 print("  /new                 - Start a new conversation thread")
                 print("  /help                - Show this help message")
                 print("  /quit                - Exit the client")

            else:
                # Send as chat message
                if not current_conversation_id:
                    print("[!] No active conversation. Use '/new' to start one.")
                    continue
                if not is_connected:
                    print("[!] Not connected. Please wait or try restarting.")
                    continue

                message_data = {
                    'message': line,
                    'conversation_id': current_conversation_id
                }
                await sio.emit('send_message', message_data)
                # Don't print prompt here, wait for response_complete

        except socketio.exceptions.ConnectionError as e:
             print(f"\n[!] Connection error occurred: {e}. Attempting to reconnect...")
             is_connected = False
             await asyncio.sleep(5)
        except KeyboardInterrupt:
             print("\n[*] Caught interrupt, disconnecting...")
             if is_connected:
                 await sio.disconnect()
             break
        except Exception as e:
             print(f"\n[!] An unexpected error occurred: {e}")
             # Decide whether to break or try to continue
             await asyncio.sleep(1) # Avoid tight loop on error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="WebSocket Chat Client")
    parser.add_argument(
        "--url", default=DEFAULT_SERVER_URL, help=f"URL of the backend server (default: {DEFAULT_SERVER_URL})"
    )
    args = parser.parse_args()

    print("--- Simple WebSocket Chat Client ---")
    print(f"--- Connecting to: {args.url} ---")
    print("--- Type /help for commands ---")

    try:
        asyncio.run(main_loop(args.url))
    except KeyboardInterrupt:
        print("\n[*] Exiting client.")
    finally:
         # Ensure disconnect on exit if loop breaks unexpectedly
         if is_connected:
              # Run disconnect in a new loop if the main one is closing
              try:
                  asyncio.run(sio.disconnect())
              except RuntimeError: # Loop already closed
                  pass
         print("[*] Client shutdown complete.") 