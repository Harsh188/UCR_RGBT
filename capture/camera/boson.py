from camera.camera import Camera
from flirpy.camera.boson import Boson
import queue
import threading
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BosonCamera(Camera):
    def __init__(self):
        self.camera = Boson()
        self._set_external_sync()
        self.frame_buffer = queue.Queue(maxsize=500)
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        self.clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def _set_external_sync(self):
        # Set external sync mode
        self.camera.set_external_sync_mode(1)
        print(f"Boson Sync Mode: {self.camera.get_external_sync_mode()}")

    def _capture_frames(self):
        while not self.stop_event.is_set():
            frame = self.camera.grab()
            if frame is not None:
                # Flip the frame vertically (across the horizontal axis)
                flipped_frame = cv2.flip(frame, 1)
                if self.frame_buffer.full():
                    self.frame_buffer.get() # Remove oldest frame if buffer is full
                self.frame_buffer.put(flipped_frame)
            else:
                logging.warning("Failed to capture Boson frame")

    def capture_frame(self):
        try:
            return self.frame_buffer.get_nowait()
        except queue.Empty:
            return None

    def get_clache_frame(self, frame):
        clache = self.clache.apply(frame)
        return clache

    def close(self):
        self.stop_event.set()
        self.capture_thread.join()
        self.camera.close()