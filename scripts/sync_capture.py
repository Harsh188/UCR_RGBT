import cv2
import numpy as np
import os
import json
from datetime import datetime
from abc import ABC, abstractmethod
from flirpy.camera.boson import Boson
import PySpin
import argparse

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
            print("Failed to capture Boson frame")
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
        self.camera.BeginAcquisition()

    def capture_frame(self):
        image_result = self.camera.GetNextImage()
        if image_result.IsIncomplete():
            print(f"Blackfly image incomplete with image status {image_result.GetImageStatus()}")
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

class SceneRecorder:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.boson_camera = BosonCamera()
        self.blackfly_camera = BlackflyCamera()
        self.scene_number = self.get_next_scene_number()

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

    @staticmethod
    def save_frame(frame, directory, frame_number):
        filename = os.path.join(directory, f"frame_{frame_number:06d}.png")
        cv2.imwrite(filename, frame)

    @staticmethod
    def save_meta(meta, directory, frame_number):
        filename = os.path.join(directory, f"meta_{frame_number:06d}.json")
        with open(filename, 'w') as f:
            json.dump(meta, f)

    def capture_scene(self, scene_dir, max_frames=1000):
        frame_number = 0
        while frame_number < max_frames:
            boson_frame = self.boson_camera.capture_frame()
            blackfly_frame = self.blackfly_camera.capture_frame()

            if boson_frame is None or blackfly_frame is None:
                continue

            self.save_frame(boson_frame, os.path.join(scene_dir, "lwir", "raw", "data"), frame_number)
            
            boson_agc_rgb = self.boson_camera.get_agc_frame(boson_frame)
            self.save_frame(boson_agc_rgb, os.path.join(scene_dir, "lwir", "agc", "data"), frame_number)
            
            self.save_frame(blackfly_frame, os.path.join(scene_dir, "rgb", "data"), frame_number)
            
            meta = self.calculate_meta(frame_number, boson_frame, blackfly_frame)
            self.save_meta(meta, os.path.join(scene_dir, "lwir", "raw", "meta"), frame_number)
            self.save_meta(meta, os.path.join(scene_dir, "lwir", "agc", "meta"), frame_number)
            self.save_meta(meta, os.path.join(scene_dir, "rgb", "meta"), frame_number)

            print(f"Saved frame {frame_number}")
            frame_number += 1

            cv2.imshow('Boson', boson_agc_rgb)
            cv2.imshow('Blackfly', blackfly_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return frame_number

    @staticmethod
    def calculate_meta(frame_number, boson_frame, blackfly_frame):
        return {
            "timestamp": datetime.now().isoformat(),
            "frame_number": frame_number,
            "boson_min": int(np.min(boson_frame)),
            "boson_max": int(np.max(boson_frame)),
            "boson_mean": float(np.mean(boson_frame)),
            "blackfly_mean": float(np.mean(blackfly_frame))
        }

    def record_scenes(self):
        try:
            while True:
                scene_dir = self.create_directory_structure()
                print(f"Recording scene {self.scene_number}. Press 'q' to stop recording.")
                frames_captured = self.capture_scene(scene_dir)
                print(f"Scene {self.scene_number} completed. {frames_captured} frames captured.")

                user_input = input("Press 'q' to quit or any other key to record another scene: ")
                if user_input.lower() == 'q':
                    break
                self.scene_number = self.get_next_scene_number()
        finally:
            self.close_cameras()

    def close_cameras(self):
        self.boson_camera.close()
        self.blackfly_camera.close()
        cv2.destroyAllWindows()

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