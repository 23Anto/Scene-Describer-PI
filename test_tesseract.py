import cv2
import numpy as np
import requests
import pytesseract
import pyttsx3
import threading
import time


URL = "http://192.168.4.1/stream"


# -----------------------------
# Text-to-speech setup
# -----------------------------

speech_queue = []
speech_lock = threading.Lock()


def speak_worker():
    engine = pyttsx3.init()

    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    while True:

        text = None

        with speech_lock:
            if speech_queue:
                text = speech_queue.pop(0)

        if text:
            print("Speaking:", text)
            engine.say(text)
            engine.runAndWait()
        else:
            time.sleep(0.1)


def start_speech_thread():

    speech_thread = threading.Thread(
        target=speak_worker,
        daemon=True
    )

    speech_thread.start()

    return speech_thread


def speak(text):

    with speech_lock:
        speech_queue.append(text)


# -----------------------------
# OCR process
# -----------------------------

def ocr_process(cancel_event):

    print("Starting OCR...")

    # Start TTS when OCR starts
    start_speech_thread()

    response = None

    try:

        # -----------------------------
        # Connect to ESP32-CAM
        # -----------------------------

        print("Connecting to camera...")

        response = requests.get(
            URL,
            stream=True,
            timeout=10
        )

        print("Status:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))


        # -----------------------------
        # MJPEG frame processing
        # -----------------------------

        buffer = b""

        last_ocr_time = 0
        ocr_interval = 2.0

        last_text = ""


        # --------------------------------
        # Keep running until Vosk says stop
        # --------------------------------

        while not cancel_event.is_set():

            chunk = response.raw.read(4096)

            if not chunk:
                print("Stream ended.")
                break

            buffer += chunk


            while True:

                # Check for "stop"
                if cancel_event.is_set():
                    break


                # -----------------------------
                # Find beginning of JPEG
                # -----------------------------

                start = buffer.find(b"\xff\xd8")

                if start == -1:
                    buffer = buffer[-2:]
                    break


                # -----------------------------
                # Find end of JPEG
                # -----------------------------

                end = buffer.find(
                    b"\xff\xd9",
                    start + 2
                )

                if end == -1:
                    break


                jpg = buffer[start:end + 2]

                buffer = buffer[end + 2:]


                # -----------------------------
                # Decode JPEG
                # -----------------------------

                frame = cv2.imdecode(
                    np.frombuffer(
                        jpg,
                        dtype=np.uint8
                    ),
                    cv2.IMREAD_COLOR
                )

                if frame is None:
                    continue


                # -----------------------------
                # Display camera
                # -----------------------------



                # -----------------------------
                # OCR every 2 seconds
                # -----------------------------

                current_time = time.time()

                if current_time - last_ocr_time >= ocr_interval:

                    last_ocr_time = current_time

                    print("Running OCR...")

                    text = pytesseract.image_to_string(frame)

                    # Clean up OCR result
                    text = " ".join(text.split())



                    print("Detected:", text)


                    # Only speak if text changed
                    if text != last_text:
                    	speak(text)

                    	last_text = text


            # Stop if Vosk received "stop"
            if cancel_event.is_set():
                break


    except requests.exceptions.RequestException as e:

        print(f"Camera connection error: {e}")


    except Exception as e:

        print(f"OCR error: {e}")


    finally:

        # -----------------------------
        # Cleanup
        # -----------------------------

        if response is not None:
            response.close()

        cv2.destroyAllWindows()

        print("OCR stopped.")
