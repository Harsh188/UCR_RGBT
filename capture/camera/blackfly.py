# Imports
import PySpin
from camera.camera import Camera
import queue
import threading
import cv2
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXPOSURE_TIME = 10000 # in microseconds
GAIN_VALUE = 0 #in dB, 0-40;
GAMMA_VALUE = 0.5 #0.25-1
SEC_TO_RECORD = 10 #approximate # seconds to record for; can also use Ctrl-C to interupt in middle of capture
IMAGE_HEIGHT = 540  #540 pixels default
IMAGE_WIDTH = 720 #720 pixels default
HEIGHT_OFFSET = round((IMAGE_HEIGHT)/2) # Y, to keep in middle of sensor
WIDTH_OFFSET = round((IMAGE_WIDTH)/2) # X, to keep in middle of sensor

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
        self.camera.AcquisitionFrameRateEnable.SetValue(False)
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

        # if not self.set_external_sync_mode():
        #     raise Exception("Unable to set syncronization.")
        
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

    def set_external_sync_mode(self):
        '''
        Sets the camera to external sync mode.
        '''
        try:
            # Ensure camera is initialized
            if not self.camera.IsInitialized():
                return False

            # Get node for trigger
            nodemap = self.camera.GetNodeMap()
            trigger_mode_node = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))
            if not PySpin.IsAvailable(trigger_mode_node) or not PySpin.IsWritable(trigger_mode_node):
                print("Unable to access TriggerMode node.")
                return False
            
            # Set triger mode to "Off"
            self.camera.TriggerMode.SetValue(PySpin.TriggerMode_Off)

            # Set trigger source to Line3 (adjust if using a different input line)
            self.camera.TriggerSource.SetValue(PySpin.TriggerSource_Line0)

            # Set trigger overlap to ReadOut for better performance
            self.camera.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
            
            # Enable trigger mode
            self.camera.TriggerMode.SetValue(PySpin.TriggerMode_On)
            return True
        
        except PySpin.SpinnakerException as ex:
            print("Error setting external sync mode: %s" % ex)
            return False
    
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