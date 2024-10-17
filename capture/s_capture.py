import PySpin
import flirpy
from flirpy.camera.boson import Boson
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_blackfly(cam):
    nodemap = cam.GetNodeMap()
    set_software_sync_mode(cam, nodemap)

    # Set up other parameters
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(59.0)
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureTime.SetValue(30000.0)
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(10.0)
    cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

def set_software_sync_mode(cam, nodemap):
    '''
    Sets up the camera to use software trigger.
    '''
    result = True
    logging.info('*** CONFIGURING TRIGGER ***\n')

    try:
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        # Get node for trigger
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsReadable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
            logging.error('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsReadable(node_trigger_mode_off):
            logging.error('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        logging.info('Trigger mode disabled...')

        # Set TriggerSelector to FrameStart
        # For this example, the trigger selector should be set to frame start.
        # This is the default for most cameras.
        node_trigger_selector= PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
        if not PySpin.IsReadable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
            logging.error('Unable to get trigger selector (node retrieval). Aborting...')
            return False

        node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
        if not PySpin.IsReadable(node_trigger_selector_framestart):
            logging.error('Unable to set trigger selector (enum entry retrieval). Aborting...')
            return False
        node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())

        logging.info('Trigger selector set to frame start...')

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
        # mode is off.
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsReadable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            logging.error('Unable to get trigger source (node retrieval). Aborting...')
            return False
        
        node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
        if not PySpin.IsReadable(node_trigger_source_software):
            logging.error('Unable to get trigger source (enum entry retrieval). Aborting...')
            return False
        node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
        logging.info('Trigger source set to software...')

        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
        if not PySpin.IsReadable(node_trigger_mode_on):
            logging.error('Unable to enable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
        logging.info('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        logging.error('Error: %s' % ex)
        return False

    return result

def reset_trigger(nodemap):
    """
    This function returns the camera to a normal state by turning off trigger mode.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsReadable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def configure_boson(boson):
    # Set Boson to external sync slave mode
    boson.set_external_sync_mode(2)  # 2 is for slave mode

def live_visualizer(blackfly_cam, boson_cam):
    cv2.namedWindow("Combined View", cv2.WINDOW_NORMAL)

    blackfly_cam.BeginAcquisition()

    try:
        while True:
            blackfly_img, boson_img = capture_image_pair(blackfly_cam, boson_cam)
            
            if blackfly_img is None or boson_img is None:
                continue

            # Resize Blackfly image to match Boson image height
            boson_height, boson_width = boson_img.shape[:2]
            blackfly_height, blackfly_width = blackfly_img.shape[:2]
            scale_factor = boson_height / blackfly_height
            blackfly_resized = cv2.resize(blackfly_img, (int(blackfly_width * scale_factor), boson_height))

            # Normalize Boson image for display (assuming it's a thermal image)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # boson_normalized = clahe.apply(boson_img)
            boson_normalized = cv2.normalize(boson_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            boson_colormap = cv2.applyColorMap(boson_normalized, cv2.COLORMAP_INFERNO)

            # Ensure both images have 3 channels
            if len(blackfly_resized.shape) == 2:
                blackfly_resized = cv2.cvtColor(blackfly_resized, cv2.COLOR_GRAY2BGR)
            if len(boson_colormap.shape) == 2:
                boson_colormap = cv2.cvtColor(boson_colormap, cv2.COLOR_GRAY2BGR)

            # Create a combined image
            combined_img = np.hstack((blackfly_resized, boson_colormap))

            # Display combined image
            cv2.imshow("Combined View", combined_img)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        blackfly_cam.EndAcquisition()
        cv2.destroyAllWindows()

def capture_image_pair(blackfly_cam, boson_cam):
    try:
        # Execute software trigger for Blackfly
        nodemap = blackfly_cam.GetNodeMap()
        node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
        if not PySpin.IsWritable(node_softwaretrigger_cmd):
            print('Unable to execute trigger. Aborting...')
            return None, None

        node_softwaretrigger_cmd.Execute()

        # Capture from Boson
        boson_image = boson_cam.grab()

        if boson_image is not None:
            # Flip the frame vertically (across the horizontal axis)
            boson_image = cv2.flip(boson_image, 1)

        # Capture from Blackfly
        blackfly_image = blackfly_cam.GetNextImage(1000)
        if blackfly_image.IsIncomplete():
            print("Blackfly image incomplete. Skipping.")
            blackfly_image.Release()
            return None, None
        
        # Convert Blackfly image to numpy array
        blackfly_array = blackfly_image.GetNDArray()
        blackfly_array = cv2.cvtColor(blackfly_array, cv2.COLOR_BayerRG2RGB)
        
        blackfly_image.Release()
        
        return blackfly_array, boson_image
    
    except PySpin.SpinnakerException as e:
        print(f"Error: {e}")
        return None, None

def main():
    # Initialize Blackfly camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    blackfly_cam = cam_list[0]
    blackfly_cam.Init()
    
    # Initialize Boson camera
    boson_cam = Boson()
    
    try:
        configure_blackfly(blackfly_cam)
        configure_boson(boson_cam)
        
        live_visualizer(blackfly_cam, boson_cam)
    
    finally:
        blackfly_cam.DeInit()
        del blackfly_cam
        cam_list.Clear()
        system.ReleaseInstance()
        boson_cam.close()

if __name__ == "__main__":
    main()