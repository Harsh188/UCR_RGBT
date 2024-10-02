import cv2
import numpy as np
import os
import json
from datetime import datetime
from abc import ABC, abstractmethod
from flirpy.camera.boson import Boson
import PySpin
import argparse
import threading
import queue
import signal
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Camera(ABC):
    @abstractmethod
    def capture_frame(self):
        pass

    @abstractmethod
    def close(self):
        pass

class BosonCamera(Camera):
    def __init__(self):
        self.camera = Boson()

    def capture_frame(self):
        frame = self.camera.grab()
        if frame is None:
            logging.warning("Failed to capture Boson frame")
            return None
        return frame

    def get_agc_frame(self, frame):
        agc = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(agc, cv2.COLOR_GRAY2RGB)

    def close(self):
        self.camera.close()

class BlackflyCamera(Camera):
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.camera = self.cam_list[0]
        self.camera.Init()
        self.camera.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
        self.camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.camera.BeginAcquisition()

    def capture_frame(self):
        image_result = self.camera.GetNextImage()
        if image_result.IsIncomplete():
            logging.warning(f"Blackfly image incomplete with image status {image_result.GetImageStatus()}")
            image_result.Release()
            return None
        frame = image_result.GetNDArray()
        image_result.Release()
        return frame

    def close(self):
        self.camera.EndAcquisition()
        self.camera.DeInit()
        del self.camera
        self.cam_list.Clear()
        self.system.ReleaseInstance()

class FrameProducer(threading.Thread):
    def __init__(self, camera, frame_queue, stop_event, camera_type):
        super().__init__()
        self.camera = camera
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.camera_type = camera_type

    def run(self):
        try:
            while not self.stop_event.is_set():
                frame = self.camera.capture_frame()
                if frame is not None:
                    self.frame_queue.put((self.camera_type, frame))
        except Exception as e:
            logging.error(f"FrameProducer encountered an error: {e}")
            self.stop_event.set()  # Signal the consumer to stop

class FrameConsumer(threading.Thread):
    def __init__(self, frame_queue, scene_dir, stop_event, boson_camera):
        super().__init__()
        self.frame_queue = frame_queue
        self.scene_dir = scene_dir
        self.stop_event = stop_event
        self.frame_number = 0
        self.boson_camera = boson_camera

    def run(self):
        try:
            while not self.stop_event.is_set() or not self.frame_queue.empty():
                try:
                    frame_data = self.frame_queue.get(timeout=1)
                    self.process_frame(frame_data)
                    self.frame_number += 1
                except queue.Empty:
                    continue
        except Exception as e:
            logging.error(f"FrameConsumer encountered an error: {e}")
            self.stop_event.set()  # Signal to stop

    def process_frame(self, frame_data):
        camera_type, frame = frame_data
        if camera_type == "boson":
            self.save_frame(frame, os.path.join(self.scene_dir, "lwir", "raw", "data"), self.frame_number)
            boson_agc_rgb = self.boson_camera.get_agc_frame(frame)
            self.save_frame(boson_agc_rgb, os.path.join(self.scene_dir, "lwir", "agc", "data"), self.frame_number)
        elif camera_type == "blackfly":
            self.save_frame(frame, os.path.join(self.scene_dir, "rgb", "data"), self.frame_number)

        meta = self.calculate_meta(self.frame_number, frame)
        self.save_meta(meta, os.path.join(self.scene_dir, "meta"), self.frame_number)

    @staticmethod
    def save_frame(frame, directory, frame_number):
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"frame_{frame_number:06d}.png")
        cv2.imwrite(filename, frame)

    @staticmethod
    def save_meta(meta, directory, frame_number):
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"meta_{frame_number:06d}.json")
        with open(filename, 'w') as f:
            json.dump(meta, f)

    @staticmethod
    def calculate_meta(frame_number, frame):
        return {
            "timestamp": datetime.now().isoformat(),
            "frame_number": frame_number,
            "min": int(np.min(frame)),
            "max": int(np.max(frame)),
            "mean": float(np.mean(frame))
        }

class SceneRecorder:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.boson_camera = BosonCamera()
        self.blackfly_camera = BlackflyCamera()
        self.scene_number = self.get_next_scene_number()
        self.frame_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()

    def signal_handler(self, signum, frame):
        logging.info('Received termination signal. Exiting...')
        self.stop_event.set()  # Signal threads to stop

    def get_next_scene_number(self):
        scene_number = 1
        while os.path.exists(os.path.join(self.base_dir, f'scene_{scene_number}')):
            scene_number += 1
        return scene_number

    def create_directory_structure(self):
        scene_dir = os.path.join(self.base_dir, f"scene_{self.scene_number}")
        directories = [
            os.path.join(scene_dir, "lwir", "agc", subdir)
            for subdir in ["meta", "flow", "data"]
        ] + [
            os.path.join(scene_dir, "lwir", "raw", subdir)
            for subdir in ["meta", "flow", "data"]
        ] + [
            os.path.join(scene_dir, "rgb", subdir)
            for subdir in ["meta", "flow", "data"]
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        return scene_dir

    def record_scene(self):
        scene_dir = self.create_directory_structure()
        logging.info(f"Recording scene {self.scene_number}. Press 'q' to stop recording.")

        boson_producer = FrameProducer(self.boson_camera, self.frame_queue, self.stop_event, "boson")
        blackfly_producer = FrameProducer(self.blackfly_camera, self.frame_queue, self.stop_event, "blackfly")
        consumer = FrameConsumer(self.frame_queue, scene_dir, self.stop_event, self.boson_camera)

        boson_producer.start()
        blackfly_producer.start()
        consumer.start()

        try:
            while not self.stop_event.is_set():
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("User requested stop.")
                    break
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received.")
        finally:
            self.stop_event.set()
            boson_producer.join()
            blackfly_producer.join()
            consumer.join()

        logging.info(f"Scene {self.scene_number} completed. {consumer.frame_number} frames captured.")
        self.scene_number = self.get_next_scene_number()

    def record_scenes(self):
        signal.signal(signal.SIGINT, self.signal_handler)  # Handle Ctrl+C
        try:
            while True:
                self.record_scene()
                if self.stop_event.is_set():
                    break
                user_input = input("Press 'q' to quit or any other key to record another scene: ")
                if user_input.lower() == 'q':
                    break
        finally:
            self.close_cameras()

    def close_cameras(self):
        logging.info("Closing cameras and cleaning up resources.")
        self.boson_camera.close()
        self.blackfly_camera.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Scene Recorder for Boson and Blackfly cameras")
    parser.add_argument("--base_dir", type=str, default="UCRT", help="Base directory for saving scenes")
    return parser.parse_args()

def main():
    args = parse_arguments()
    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)
    recorder = SceneRecorder(base_dir)
    recorder.record_scenes()

if __name__ == "__main__":
    main()