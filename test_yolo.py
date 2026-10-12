import os
import cv2
import time
import threading
import pyttsx3

from ultralytics import YOLO
from video_source import open_video_source


# ---------------------------------
# Configuration
# ---------------------------------

DIAGNOSTIC_FLOOR = 0.15
PRODUCTION_THRESHOLD = 0.7


# ---------------------------------
# Load YOLO model
# ---------------------------------

model = YOLO(
    os.environ.get(
        'YOLO_WEIGHTS',
        'yolov8n.pt'
    )
)


# ---------------------------------
# Text-to-speech
# ---------------------------------

engine = pyttsx3.init()

engine.setProperty(
    "rate",
    160
)

engine.setProperty(
    "volume",
    1.0
)


def speak(text):

    print(f"[YOLO] Speaking: {text}")

    engine.say(text)
    engine.runAndWait()


# ---------------------------------
# YOLO process
# ---------------------------------

def yolo_process(cancel_event):

    cap = None

    # Objects detected during the
    # previous YOLO inference
    previous_objects = set()

    # Holds ONLY the newest camera frame
    latest_frame = [None]

    # Protects latest_frame
    frame_lock = threading.Lock()

    try:

        # ---------------------------------
        # Connect to camera
        # ---------------------------------

        print("[YOLO] Connecting to camera...")

        cap = open_video_source()

        if not cap or not cap.isOpened():

            print(
                "[YOLO Error] "
                "Could not open video source."
            )

            return


        print("[YOLO] Camera connected.")


        # ---------------------------------
        # Camera reader thread
        # ---------------------------------

        def camera_reader():

            while not cancel_event.is_set():

                ok, frame = cap.read()

                if not ok:
                    continue

                # Always replace the old frame.
                # We NEVER allow a queue of old frames.
                with frame_lock:

                    latest_frame[0] = frame


        camera_thread = threading.Thread(
            target=camera_reader,
            daemon=True
        )

        camera_thread.start()


        print(
            "[YOLO] Object detection started."
        )


        # ---------------------------------
        # YOLO processing loop
        # ---------------------------------

        while not cancel_event.is_set():

            # Get the newest available frame
            with frame_lock:

                if latest_frame[0] is None:

                    frame = None

                else:

                    frame = latest_frame[0].copy()


            # Camera hasn't provided a frame yet
            if frame is None:

                time.sleep(0.01)
                continue


            # Check for stop command
            if cancel_event.is_set():
                break

            # ---------------------------------
            # Run YOLO on newest frame
            # ---------------------------------

            results = model(
                frame,
                verbose=False,
                conf=DIAGNOSTIC_FLOOR,
                device="cpu"
            )


            # ---------------------------------
            # Find detected objects
            # ---------------------------------

            current_objects = set()


            for box in results[0].boxes:

                score = float(box.conf[0])


                # Ignore low-confidence detections
                if score < PRODUCTION_THRESHOLD:
                    continue


                class_id = int(
                    box.cls[0]
                )


                label = model.names[
                    class_id
                ]


                current_objects.add(
                    label
                )


            # ---------------------------------
            # Find newly appearing objects
            # ---------------------------------

            newly_appeared = (
                current_objects
                - previous_objects
            )


            # ---------------------------------
            # Announce new objects
            # ---------------------------------

            for label in newly_appeared:

                if cancel_event.is_set():
                    break

                print(
                    f"[YOLO] New object: {label}"
                )

                speak(label)


            # Remember objects from this frame
            previous_objects = current_objects


        print(
            "[YOLO] Stop command received."
        )


    except Exception as e:

        print(
            f"[YOLO Error]: {e}"
        )


    finally:

        # ---------------------------------
        # Cleanup
        # ---------------------------------

        if cap is not None:

            try:

                cap.release()

                print(
                    "[YOLO] Network stream closed."
                )

            except Exception:
                pass


        cv2.destroyAllWindows()


        print(
            "[YOLO] Detection stopped."
        )
