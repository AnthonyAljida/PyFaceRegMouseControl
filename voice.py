import speech_recognition as sr
import threading
import webbrowser
import pyautogui

# Define a global variable to store the recognized command
recognized_command = ""


# Function to continuously listen for voice commands
def listen_for_commands():
    global recognized_command
    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            print("Listening for voice commands...")
            recognizer.adjust_for_ambient_noise(
                source, duration=1.5
            )  # Adjust for ambient noise
            audio = recognizer.listen(source)  # Limit recording to 3 seconds

        try:
            command = recognizer.recognize_google(audio)
            print("Recognized command:", command)
            # Store recognized command in lowercase
            recognized_command = command.lower()

            #  Check the command
            if "chrome" in recognized_command:
                chrome()
            elif "browser" in recognized_command:
                close_browser()
            elif "tab" in recognized_command:
                close_tab()
            elif "type" in recognized_command:
                type(recognized_command)
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the command.")
        except sr.RequestError as e:
            print(f"Error retrieving speech recognition results: {e}")


# Function to open Google Chrome
def chrome():
    print("Opening Google Chrome...")
    webbrowser.open("https://www.google.com")


# Simulate key combination to close the active window (browser)
def close_browser():
    print("Closing Google Chrome...")
    pyautogui.hotkey("alt", "f4")  # Simulate pressing Alt + F4 keys


def close_tab():
    print("Closing Google Chrome...")
    pyautogui.hotkey("ctrl", "w")  # Simulate pressing Alt + F4 keys


def type(recognized_command):

    words_to_type = " ".join(recognized_command.split()[1:])
    print(words_to_type)
    pyautogui.write(words_to_type)


# Function to start voice command recognition in a separate thread
def start_voice_recognition():
    voice_thread = threading.Thread(target=listen_for_commands)
    voice_thread.daemon = (
        True  # Daemonize the thread to automatically terminate with the main program
    )
    voice_thread.start()
