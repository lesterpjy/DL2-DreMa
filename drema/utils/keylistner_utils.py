from pynput import keyboard
import threading

# Function to handle key press events
pressed_key = None
def on_press(key):
    print("Key pressed: {0}".format(key))
    global pressed_key

    try:
        pressed_key = key.char  # For alphanumeric keys
    except AttributeError:
        pressed_key = str(key)  # For special keys (e.g., Key.esc)

# Start listening for keyboard events in a non-blocking way
def listen_keys():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def start_listener():
    """
    Starts the key listener in a separate daemon thread.
    """
    listener_thread = threading.Thread(target=listen_keys, daemon=True)
    listener_thread.start()

def get_pressed_key():
    """
    Returns the last pressed key and clears it.
    """
    global pressed_key
    key = pressed_key
    pressed_key = None  # Clear after reading
    return key
