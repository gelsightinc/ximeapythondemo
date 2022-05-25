from ximea import xiapi
import cv2
import numpy as np
import os
import argparse
#from cv_bridge import CvBridge, CvBridgeError
import configparser
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow
from sklearn.decomposition import PCA
from tempfile import TemporaryFile
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import glob

#import plantcv as pcv

backSub = cv2.createBackgroundSubtractorMOG2()


# Settings
CAMERAFPS       = 30
RESCALE         = 3
STDDEVTHRESH    = 1.5

# Global variables for background subtraction
backidx         = 0
nbackground     = 10
subtractback    = False
backgroundinit  = False

'''
This script reads the ximea camera (GS R1 sensor) using 
the ximea api and saves them as either video or images
'''

class GelSight:
    def __init__(self):
        # create a publisher to publish raw image
        # self.bridge = CvBridge()

        #variable to store data
        self.data = None

        # create handle and open self.camera
        self.cam = xiapi.Camera(dev_id=0)
        self.cam.open_device()

        self.cam.set_gpo_selector('XI_GPO_PORT1')
        self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT')
        self.cam.set_gpi_mode('XI_GPI_OFF')
        self.cam.set_imgdataformat('XI_MONO8')
        self.cam.set_gpo_mode('XI_GPO_EXPOSURE_ACTIVE')
        # self.cam.set_downsampling_type('XI_SKIPPING')
        # self.cam.set_downsampling('XI_DWN_2x2')
        #self.cam.set_manual_wb(1)

        self.cam.set_framerate(CAMERAFPS)

        self.cam.set_exposure(600) ## microseconds


        # start data acquisition
        self.cam.start_acquisition()
        # create image handle
        self.img = xiapi.Image()

    def get_image(self):
        self.cam.get_image(self.img)
        self.data = self.img.get_image_data_numpy()
        return self.data

    def end_process(self):
        # clear handle and close window
        self.cam.stop_acquisition()
        self.cam.close_device()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    gsa = GelSight()
    gsa.cam.get_image(gsa.img)
    f0 = gsa.img.get_image_data_numpy()

    imgh = int(f0.shape[0] / RESCALE)
    imgw = int(f0.shape[1] / RESCALE)

    # Buffers for background image
    avgimg = np.zeros((imgh, imgw), dtype="uint8")
    avgsq  = np.zeros((imgh, imgw), dtype="uint8")
    stdim  = np.zeros((imgh, imgw), dtype="uint8")

    # Capture images until user presses 'q'
    while 1:
        gsa.cam.get_image(gsa.img)
        gsa.data = gsa.img.get_image_data_numpy()

        img = cv2.resize(gsa.data, (imgw,imgh))

        #outimg = backSub.apply(img)

        # Do background subtraction
        if subtractback:
            if backgroundinit and backidx < nbackground:
                print("accumulating background images: ", backidx)
                floatim = img.astype('float')
                scaled = floatim/nbackground
                squared = (floatim ** 2)/nbackground
                avgimg += scaled.astype('uint8')
                avgsq  += squared.astype('uint8')

                backidx += 1
            elif backgroundinit and backidx == nbackground:
                backgroundinit = False

                stdim = avgsq - avgimg**2
                stdim[stdim < 0] = 0
                stdim = np.sqrt(stdim)
            else:
                # Create binary image for pixels above or below standard deviation threshold
                dfim = np.abs(img - avgimg)
                maskinds = np.where(dfim > STDDEVTHRESH*stdim)
                maskimg = np.zeros(avgimg.shape, dtype='uint8')
                maskimg[maskinds] = 255

                # Erosion
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                maskimg = cv2.morphologyEx(maskimg, cv2.MORPH_OPEN, kernel)

                img = maskimg

        cv2.imshow('', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("user pressed ", key)
            break
        elif key == ord('b'):
            print("user pressed ", key)
            # If background subtraction is off, capture nbackground frames
            if not subtractback:
                avgimg = np.zeros((imgh, imgw), dtype="uint8")
                avgstd = np.zeros((imgh, imgw), dtype="uint8")
                backidx = 0
                backgroundinit = True
                subtractback = True
                print("starting background subtraction")
            else:
                subtractback = False


    # cleanly shut down the camera
    gsa.end_process()
