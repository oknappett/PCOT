import cv2 as cv
import numpy as np

from pcot.datum import Datum
from pcot.xform import xformtype, XFormType
from pcot.xforms.tabimage import TabImage


def thresh(img):
    '''threshold an image, binary threshold method used'''
    if img is None:
        return None
    else:
        copy = img.copy()
        for i in range(copy.channels):
            c = copy.img[:, :, i]
            c = (c * 255).astype(np.uint8)
            ret, t = cv.threshold(c, 180, 255, cv.THRESH_BINARY)
            copy.img[:, :, i] = t
        return copy


@xformtype
class XformPctCalibrate(XFormType):
    """Locates the pct and returns the centre pixel coordinate
    of each target circle"""

    def __init__(self):
        super().__init__("pctCalibrate", "calibration", "0.0.0")
        self.addInputConnector("", Datum.IMG)
        self.addOutputConnector("", Datum.IMG)

    def createTab(self, n, w):
        return TabImage(n, w)

    def init(self, node):
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