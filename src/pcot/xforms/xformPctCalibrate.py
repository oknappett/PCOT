import cv2 as cv
import numpy as np

import pcot
from pcot.datum import Datum
from pcot.xform import xformtype, XFormType


def thresh(img):
    # threshold an image, otsu threshold method used
    if img is None:
        return None
    else:
        copy = img.copy()
        for i in range(copy.channels):
            c = copy.img[:, :, i]
            # convert the image 0-1 range to 0-255 pixels for opencv
            c = (c * 255).astype(np.uint8)
            # blur the image with a 11x11 kernel or bilateral filter
            # bilateral seems to give better circles on pct
            blur = cv.bilateralFilter(c, 10, 50, 50)
            lowerThresh = 150
            upperThresh = 255
            ret, t = cv.threshold(blur, lowerThresh, upperThresh, cv.THRESH_OTSU)
            # set the image channel to the threshold values
            copy.img[:, :, i] = t
        return copy


def edge(img, kernel):
    # edge detector on the image -> need to do before circle detection
    k = kernel
    if img is None:
        return None
    elif type(img) is np.ndarray:
        copy = img.copy()
        copy = (copy * 255).astype(np.uint8)
        blur = cv.GaussianBlur(copy, (k, k), 0)
        edges = cv.Canny(image=blur, threshold1=100, threshold2=150)
        return edges
    else:
        copy = img.copy()
        for i in range(copy.channels):
            c = copy.img[:, :, i]
            # convert 0-1 numpy array image tp 0-256
            c = (c * 255).astype(np.uint8)
            # blur image
            blur = cv.GaussianBlur(c, (k, k), 0)
            # canny edge detection on blurred image
            edges = cv.Canny(image=blur, threshold1=100, threshold2=150)
            copy.img[:, :, i] = edges
        return copy


def HoughCircles(img):
    if img is None:
        return None
    else:
        copy = img.copy()
        print("shape 1: ", copy.shape)
        circleAmount = []
        for i in range(copy.channels):
            c = copy.img[:, :, i]
            cArray = np.array(c).astype(np.uint8)
            circles = cv.HoughCircles(cArray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            print(circles)
            if circles is not None:
                circleNo = 0
                circles = np.uint16(np.around(circles))
                for j in circles[0, :]:
                    circleNo += 1
                    x, y, r = j[0], j[1], j[2]
                    cv.circle(cArray, (x, y), r, (255, 0, 0), 2)
                    cv.circle(cArray, (x, y), 1, (255, 0, 0), 2)
                circleAmount.append(circleNo)
            copy.img[:, :, i] = cArray
        print(circleAmount, np.mean(circleAmount))
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
        return PCTCalibTab(n, w)

    def init(self, node):
        # output set to null initially
        node.out = None
        node.Kernel = 11

    def perform(self, node):
        print(node.Kernel)
        data = node.getInput(0, Datum.IMG)
        if data is None:
            out = None
        else:
            img = data.copy()
            k = node.Kernel
            edges = edge(img, k)
            circles = HoughCircles(edges)
            out = circles
        node.out = Datum(Datum.IMG, out)
        node.setOutput(0, node.out)


class PCTCalibTab(pcot.ui.tabs.Tab):
    def __init__(self, node, w):
        super().__init__(w, node, "tabpctCalibration.ui")
        self.w.KernelSize.editingFinished.connect(self.KernelChanged)
        self.nodeChanged()

    def KernelChanged(self):
        self.mark()
        self.node.Kernel = self.w.KernelSize.value()
        self.changed()

    def onNodeChanged(self):
        self.w.canvas.setMapping(self.node.mapping)
        self.w.canvas.setGraph(self.node.graph)
        self.w.canvas.setPersister(self.node)
        self.w.canvas.display(self.node.out)

