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
import time
import asyncio
import aiofiles
import psutil
import pdb

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXPOSURE_TIME = 10000 # in microseconds
GAIN_VALUE = 0 #in dB, 0-40;
GAMMA_VALUE = 0.5 #0.25-1
SEC_TO_RECORD = 10 #approximate # seconds to record for; can also use Ctrl-C to interupt in middle of capture
IMAGE_HEIGHT = 540  #540 pixels default
IMAGE_WIDTH = 720 #720 pixels default
HEIGHT_OFFSET = round((IMAGE_HEIGHT)/2) # Y, to keep in middle of sensor
WIDTH_OFFSET = round((IMAGE_WIDTH)/2) # X, to keep in middle of sensor

# New class for monitoring system resources
class SystemMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while not self.stop_event.is_set():
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            logging.info(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_usage}%")
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        self.thread.join()

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
        self.camera.set_external_sync_mode(1) # Set external sync mode
        self.frame_buffer = queue.Queue(maxsize=500)
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()

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

    def get_agc_frame(self, frame):
        agc = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(agc, cv2.COLOR_GRAY2RGB)

    def close(self):
        self.stop_event.set()
        self.capture_thread.join()
        self.camera.close()

class BlackflyCamera(Camera):
    def __init__(self):
        
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.camera = self.cam_list[0]
        self.camera.Init()
        self.camera.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.camera.UserSetLoad()
        self.camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        # self.camera.ExposureAuto.SetValue(PySpin.ExposureAuto_On)
        # self.camera.ExposureTime.SetValue(EXPOSURE_TIME)
        # self.camera.AcquisitionFrameRateEnable.SetValue(False)
        # self.camera.GainAuto.SetValue(PySpin.GainAuto_Off)
        # self.camera.Gain.SetValue(GAIN_VALUE)
        # self.camera.GammaEnable.SetValue(True)
        self.camera.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)
        # self.camera.Width.SetValue(IMAGE_WIDTH)
        # self.camera.Height.SetValue(IMAGE_HEIGHT)
        # self.camera.OffsetX.SetValue(WIDTH_OFFSET)
        # self.camera.OffsetY.SetValue(HEIGHT_OFFSET)

        camTransferLayerStream = self.camera.GetTLStreamNodeMap()
        handling_mode1 = PySpin.CEnumerationPtr(camTransferLayerStream.GetNode('StreamBufferHandlingMode'))
        handling_mode_entry = handling_mode1.GetEntryByName('OldestFirst')
        handling_mode1.SetIntValue(handling_mode_entry.GetValue())
        self.camera.LineSelector.SetValue(PySpin.LineSelector_Line1)
        self.camera.LineMode.SetValue(PySpin.LineMode_Output) 
        self.camera.LineSource.SetValue(PySpin.LineSource_ExposureActive)
        

        # self.set_camera_resolution()
        # self.set_optimal_frame_rate()
        # self.optimize_camera_settings()
        frameRate = self.camera.AcquisitionResultingFrameRate()
        print('frame rate = {:.2f} FPS'.format(frameRate))
        numImages = round(frameRate*SEC_TO_RECORD)
        print('# frames = {:d}'.format(numImages))

        # self.set_buffer_handling_mode()
        self.camera.BeginAcquisition()
        self.processor = PySpin.ImageProcessor()
        # self.processor.SetColorProcessing(PySpin.HQ_LINEAR) # EXPERIMENTAL
        self.processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        self.frame_buffer = queue.Queue(maxsize=500)
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
    
    def optimize_camera_settings(self):
        self.camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.camera.ExposureTime.SetValue(1000)  # exposure time, adjust as needed

        # Optimize packet size for GigE cameras (adjust for USB3 if needed)
        self.camera.GevSCPSPacketSize.SetValue(9000)
        self.camera.GevSCPD.SetValue(0)  # Set packet delay to 0

        logging.info("Camera settings optimized for higher frame rate")

    def set_optimal_frame_rate(self):
        try:
            max_frame_rate = self.camera.AcquisitionFrameRate.GetMax()
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            # pdb.set_trace()
            self.camera.AcquisitionFrameRate.SetValue(max_frame_rate)
            logging.info(f"Frame rate set to {max_frame_rate:.2f} fps")
        except PySpin.SpinnakerException as e:
            logging.warning(f"Unable to set frame rate: {e}")
            logging.info("Continuing with default frame rate")

    def set_buffer_handling_mode(self):
        sNodemap = self.camera.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        node_newestonly_mode = node_newestonly.GetValue()
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
    
    def set_camera_resolution(self):
        # Get the maximum width and height
        max_width = self.camera.WidthMax.GetValue()
        max_height = self.camera.HeightMax.GetValue()
        
        # Set the desired resolution (e.g., half of the maximum)
        desired_width = max_width // 2
        desired_height = max_height // 2
        
        # Ensure the desired dimensions are multiples of 4
        desired_width = desired_width - (desired_width % 4)
        desired_height = desired_height - (desired_height % 4)
        
        # Set the new width and height
        self.camera.Width.SetValue(desired_width)
        self.camera.Height.SetValue(desired_height)
        
        # Center the ROI
        offset_x = (max_width - desired_width) // 2
        offset_y = (max_height - desired_height) // 2
        self.camera.OffsetX.SetValue(offset_x)
        self.camera.OffsetY.SetValue(offset_y)
        
        logging.info(f"Camera resolution set to {desired_width}x{desired_height}")

    def _capture_frames(self):
        while True:
            image_result = self.camera.GetNextImage()
            if image_result.IsIncomplete():
                image_result.Release()
                continue
            
            bayer_image = image_result.GetNDArray()
            frame = cv2.cvtColor(bayer_image, cv2.COLOR_BayerRG2RGB)

            # pdb.set_trace()
            image_result.Release()
            
            if self.frame_buffer.full():
                self.frame_buffer.get() # Remove oldest frame if buffer is full
            self.frame_buffer.put(frame)

    def reduce_resolution(self, frame):
        height, width = frame.shape[:2]
        new_height, new_width = height // 2, width // 2
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def capture_frame(self):
        try:
            return self.frame_buffer.get_nowait()
        except queue.Empty:
            return None

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
        frames_processed = 0
        start_time = time.time()
        last_log_time = start_time
        try:
            while not self.stop_event.is_set():
                frame = self.camera.capture_frame()
                if frame is not None:
                    if self.frame_queue.qsize() < 450:  # Drop frames if queue is too full
                        self.frame_queue.put(frame)
                    frames_processed += 1
                    current_time = time.time()
                    if current_time - last_log_time >= 2:  # Log every 5 seconds
                        elapsed_time = current_time - last_log_time
                        fps = frames_processed / elapsed_time
                        logging.info(f"{self.camera_type} frame rate: {fps:.2f} FPS")
                        logging.info(f"{self.camera_type} queue size: {self.frame_queue.qsize()}")
                        last_log_time = current_time
                        frames_processed = 0
        except Exception as e:
            logging.error(f"FrameProducer encountered an error: {e}", exc_info=True)
            self.stop_event.set()  # Signal the consumer to stop

class FrameConsumer(threading.Thread):
    def __init__(self, rgb_queue, lwir_queue, scene_dir, stop_event, boson_camera):
        super().__init__()
        self.rgb_queue = rgb_queue
        self.lwir_queue = lwir_queue
        self.scene_dir = scene_dir
        self.stop_event = stop_event
        self.frame_number = 0
        self.boson_camera = boson_camera
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def run(self):
        try:
            while not self.stop_event.is_set() or not self.rgb_queue.empty() or not self.lwir_queue.empty():
                try:
                    rgb_frame = self.rgb_queue.get(timeout=0.1)
                    lwir_frame = self.lwir_queue.get(timeout=0.1)
                    self.loop.run_until_complete(self.process_frame_pair(rgb_frame, lwir_frame))
                    self.frame_number += 1
                except queue.Empty:
                    continue
        except Exception as e:
            logging.error(f"FrameConsumer encountered an error: {e}", exc_info=True)
            self.stop_event.set() # Signal to stop
        finally:
            self.loop.close()

    async def process_frame_pair(self, rgb_frame, lwir_frame):
        timestamp = datetime.now()
        tasks = [
            self.save_frame(lwir_frame, os.path.join(self.scene_dir, "lwir", "raw", "data"), "LWIR_RAW", timestamp),
            self.save_frame(self.boson_camera.get_agc_frame(lwir_frame), os.path.join(self.scene_dir, "lwir", "agc", "data"), "LWIR_AGC", timestamp),
            self.save_meta(self.calculate_meta(lwir_frame), os.path.join(self.scene_dir, "lwir", "raw", "meta"), timestamp),
            self.save_meta(self.calculate_meta(self.boson_camera.get_agc_frame(lwir_frame)), os.path.join(self.scene_dir, "lwir", "agc", "meta"), timestamp),
            self.save_frame(rgb_frame, os.path.join(self.scene_dir, "rgb", "data"), "RGB", timestamp),
            self.save_meta(self.calculate_meta(rgb_frame), os.path.join(self.scene_dir, "rgb", "meta"), timestamp)
        ]
        await asyncio.gather(*tasks)

    async def save_frame(self, frame, directory, prefix, timestamp):
        os.makedirs(directory, exist_ok=True)
        filename = f"{prefix}_{self.frame_number:06d}_{timestamp.strftime('%H_%M_%S_%f')[:-3]}.png"
        _, img_encoded = cv2.imencode('.png', frame)
        async with aiofiles.open(os.path.join(directory, filename), mode='wb') as f:
            await f.write(img_encoded.tobytes())

    async def save_meta(self, meta, directory, timestamp):
        os.makedirs(directory, exist_ok=True)
        filename = f"meta_{self.frame_number:06d}_{timestamp.strftime('%H_%M_%S_%f')[:-3]}.json"
        async with aiofiles.open(os.path.join(directory, filename), mode='w') as f:
            await f.write(json.dumps(meta))

    def calculate_meta(self, frame):
        return {
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.frame_number,
            "min": int(np.min(frame)),
            "max": int(np.max(frame)),
            "mean": float(np.mean(frame))
        }

class SceneRecorder:
    def __init__(self, base_dir, recording_duration=10):
        self.base_dir = base_dir
        self.boson_camera = BosonCamera()
        self.blackfly_camera = BlackflyCamera()
        self.scene_number = self.get_next_scene_number()
        self.rgb_queue = queue.Queue(maxsize=500)
        self.lwir_queue = queue.Queue(maxsize=500)
        self.stop_event = threading.Event()
        self.recording_duration = recording_duration
        self.system_monitor = SystemMonitor()  # Initialize the system monitor

    def signal_handler(self, signum, frame):
        logging.info('Received termination signal. Exiting...')
        self.stop_event.set()  # Signal threads to stop
        self.system_monitor.stop()  # Stop the system monitor

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
        self.stop_event.clear()
        scene_dir = self.create_directory_structure()
        logging.info(f"Recording scene {self.scene_number}. Press 'q' and Enter to stop recording.")

        rgb_producer = FrameProducer(self.blackfly_camera, self.rgb_queue, self.stop_event, "rgb")
        lwir_producer = FrameProducer(self.boson_camera, self.lwir_queue, self.stop_event, "lwir")
        consumer = FrameConsumer(self.rgb_queue, self.lwir_queue, scene_dir, self.stop_event, self.boson_camera)

        rgb_producer.start()
        lwir_producer.start()
        consumer.start()

        start_time = time.time()
        try:
            while not self.stop_event.is_set():
                if time.time() - start_time > self.recording_duration:
                    logging.info(f"Recording duration of {self.recording_duration} seconds reached.")
                    break
                if input().lower() == 'q':
                    logging.info("User requested stop.")
                    break
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received.")
        finally:
            self.stop_event.set()
            # Clear the queues
            while not self.rgb_queue.empty():
                self.rgb_queue.get()
            while not self.lwir_queue.empty():
                self.lwir_queue.get()
            
            # Join threads with a timeout
            rgb_producer.join(timeout=5)
            lwir_producer.join(timeout=5)
            consumer.join(timeout=5)
            
            if rgb_producer.is_alive() or lwir_producer.is_alive() or consumer.is_alive():
                logging.warning("Some threads did not terminate properly.")
            
            logging.info(f"Scene {self.scene_number} completed. {consumer.frame_number} frames captured.")
            self.scene_number = self.get_next_scene_number()

    def record_scenes(self):
        signal.signal(signal.SIGINT, self.signal_handler)  # Handle Ctrl+C
        try:
            while True:
                self.record_scene()
                user_input = input("Press Enter to record another scene or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
        finally:
            self.close_cameras()
            self.system_monitor.stop()  # Stop the system monitor

    def close_cameras(self):
        logging.info("Closing cameras and cleaning up resources.")
        self.boson_camera.close()
        self.blackfly_camera.close()
        self.system_monitor.stop()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Scene Recorder for Boson and Blackfly cameras")
    parser.add_argument("--base_dir", type=str, default=r"C:\Users\EndUser\Documents\Programming\BosonChecks\UCR_RGBT\dataset\UCRT", help="Base directory for saving scenes")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration for each scene in seconds")
    return parser.parse_args()

def main():
    args = parse_arguments()
    base_dir = args.base_dir
    recording_duration = args.duration
    os.makedirs(base_dir, exist_ok=True)
    recorder = SceneRecorder(base_dir, recording_duration)
    recorder.record_scenes()

if __name__ == "__main__":
    main()