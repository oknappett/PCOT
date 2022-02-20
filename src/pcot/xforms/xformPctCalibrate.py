import cv2 as cv
import numpy as np

from pcot.datum import Datum
from pcot.xform import xformtype, XFormType
from pcot.xforms.tabimage import TabImage


def thresh(img):
    # threshold an image, binary threshold method used
    if img is None:
        return None
    else:
        copy = img.copy()
        for i in range(copy.channels):
            c = copy.img[:, :, i]
            # convert the image 0-1 range to 0-255 pixels for opencv
            c = (c * 255).astype(np.uint8)
            # blur the image with a 5x5 kernel
            blur = cv.GaussianBlur(c, (11,11), 0)
            lowerThresh = 180
            upperThresh = 255
            ret, t = cv.threshold(blur, lowerThresh, upperThresh, cv.THRESH_OTSU)
            # set the image channel to the threshold values
            copy.img[:, :, i] = t
        return copy


@xformtype
class XformPctCalibrate(XFormType):
    """Locates the pct and returns the centre pixel coordinate
    of each target circle"""

    def __init__(self):
        super().__init__("pctCalibrate", "calibration", "0.0.0")
        # single input image
        self.addInputConnector("", Datum.IMG)
        # single output, it's an image for now but will need to decide on a
        # output type at a later date -> the centre pixels of the targets
        self.addOutputConnector("", Datum.IMG)

    def createTab(self, n, w):
        # standard tab for now, just outputs the image after the
        # xform is applied within the node
        return TabImage(n, w)

    def init(self, node):
        # output set to null initially
        node.out = None

    def perform(self, node):
        data = node.getInput(0, Datum.IMG)
        if data is None:
            out = None
        else:
            img = data.copy()
            threshed = thresh(img)
            out = threshed
        node.out = Datum(Datum.IMG, out)
        node.setOutput(0, node.out)
