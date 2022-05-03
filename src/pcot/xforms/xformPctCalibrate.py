import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import pcot
from pcot.datum import Datum
from pcot.xform import xformtype, XFormType


class XformPctCalibrate(XFormType):
    """Class for the detection of the PCT in an image, does the perform for the detection"""

    def __init__(self, name, group, version):
        super().__init__(name, group, version)
        # single input image
        self.addInputConnector("input", Datum.IMG)
        # single output, it's an image for now but will need to decide on a
        # output type at a later date -> the centre pixels of the targets
        self.addOutputConnector("output", Datum.IMG)

    @staticmethod
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
                ret, t = cv.threshold(blur, lowerThresh, upperThresh, cv.THRESH_BINARY_INV)
                # set the image channel to the threshold values
                copy.img[:, :, i] = t
            return copy

    @staticmethod
    def cluster(Circles):
        if Circles is None:
            return None
        else:
            # sort circles in reverse size/radius order
            Circles = sorted(Circles, key=lambda x: x[2], reverse=True)
            copy = Circles.copy()

            clustered = []
            # loop through all circles
            for c in copy:
                x, y, r = c[0], c[1], c[2]
                SameCircles = []
                for j in reversed(copy):
                    # loop through all circles and compare to current circle
                    if j is not c:
                        Jx, Jy, Jr = j[0], j[1], j[2]
                        # if current circle centre is inside other circle then remove circle from list and add to
                        # similar circle list
                        if ((x <= (Jx + Jr)) and (x >= (Jx - Jr)) or (Jx == x)) and (
                                (y <= (Jy + Jr)) and (y >= (Jy - Jr)) or (Jy == y)):
                            SameCircles.append(j)
                            copy.remove(j)
                # copy.remove(c)
                x = []
                y = []
                r = []
                for n in SameCircles:
                    Nx, Ny, Nr = n[0], n[1], n[2]
                    x.append(Nx)
                    y.append(Ny)
                    r.append(Nr)
                xMean = np.uint16(np.around(np.mean(x)))
                yMean = np.uint16(np.around(np.mean(y)))
                rMean = np.uint16(np.around(np.mean(r)))
                if (xMean and yMean and rMean) != 0:
                    clustered.append([xMean, yMean, rMean])

            print("Targets detected: ", len(clustered))
            print("Detected target location: ", clustered)
            return clustered

    @staticmethod
    def edge(img, kernel):
        """Canny edge detector on image"""
        # edge detector on the image -> need to do before circle detection
        k = kernel
        if img is None:
            return None
        elif type(img) is np.ndarray:
            copy = img.copy()
            # convert image 0-1 to 0-256 for opencv to work. 2^8 = 8bit image depth
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

    @staticmethod
    def drawCircles(img, Circles):
        if Circles is None:
            return None
        else:
            copy = img.copy()
            rgb = copy.rgbImage()
            for n in Circles:
                x, y, r = n[0], n[1], n[2]
                cv.circle(rgb.img, (x, y), r, (1, 0, 0), 1)
                cv.circle(rgb.img, (x, y), 1, (0, 1, 0), 1)
            return rgb

    def detect(self):
        pass

    def perform(self, node):
        data = node.getInput(0, Datum.IMG)
        if data is None:
            out = None
        else:
            img = data.copy()
            if node.Detect:
                circles = self.detect(img, node)
                clustered = self.cluster(circles)
                out = self.drawCircles(img, clustered)
            else:
                out = img
        node.out = Datum(Datum.IMG, out)
        node.setOutput(0, node.out)


@xformtype
class HoughCircles(XformPctCalibrate):
    """Class for Hough Circle detection"""

    def __init__(self):
        super().__init__("Hough Circles", "calibration", "0.0.0")

    def init(self, node):
        node.Kernel = 11
        node.Detect = False
        node.out = None

    def createTab(self, n, w):
        # n -> node
        # w -> window
        # standard tab for now, just outputs the image after the
        # xform is applied within the node
        return PCTHoughTab(n, w)

    def detect(self, img, node):
        if img is None:
            return None
        else:
            circleArray = []

            k = node.Kernel
            copy = img.copy()
            edges = self.edge(copy, k)

            for i in range(edges.channels):
                c = edges.img[:, :, i]

                # convert edge image to a np array of uint
                cArray = np.array(c).astype(np.uint8)

                # hough circle detection
                circles = cv.HoughCircles(cArray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0,
                                          maxRadius=0)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    # draw circles on rgb image
                    for j in circles[0, :]:
                        x, y, r = j[0], j[1], j[2]
                        circleArray.append([x, y, r])

            return circleArray


@xformtype
class BlobDetector(XformPctCalibrate):

    def __init__(self):
        super().__init__("Blob detector", "calibration", "0.0.0")

    def createTab(self, n, w):
        # n -> node
        # w -> window
        # standard tab for now, just outputs the image after the
        # xform is applied within the node
        return PCTBlobDetector(n, w)

    def init(self, node):
        node.Detect = False
        node.minThresh = 0
        node.maxThresh = 255
        node.areaFilter = False
        node.minArea = 0
        node.circularityFilter = False
        node.minCircularity = 0.0
        node.convexityFilter = False
        node.minConvexity = 0.0
        node.inertiaFilter = False
        node.minInertia = 0.0

        node.out = None

    def detect(self, img, node):
        if img is None:
            return None
        else:
            copy = img.copy()

            params = cv.SimpleBlobDetector_Params()

            # thresholds
            params.minThreshold = node.minThresh
            params.maxThreshold = node.maxThresh
            # filter by area
            params.filterByArea = node.areaFilter
            params.minArea = node.minArea

            # Filter by Circularity 
            params.filterByCircularity = node.circularityFilter
            params.minCircularity = node.minCircularity

            # Filter by Convexity 
            params.filterByConvexity = node.convexityFilter
            params.minConvexity = node.minConvexity

            # Filter by Inertia
            params.filterByInertia = node.inertiaFilter
            params.minInertiaRatio = node.minInertia

            # set up detector
            detector = cv.SimpleBlobDetector_create(params)

            # copy = self.thresh(copy)

            coords = []
            for i in range(copy.channels):
                c = copy.img[:, :, i]
                c = (c * 255).astype(np.uint8)
                c = cv.GaussianBlur(c, (5, 5), 0)
                ret, t = cv.threshold(c, 50, 255, cv.THRESH_BINARY_INV)
                keypoints = detector.detect(t)

                for k in keypoints:
                    x = k.pt[0]
                    y = k.pt[1]
                    # radius is diameter/2
                    r = k.size / 2
                    if r > 10:
                        xyr = [np.uint16(np.around(x)), np.uint16(np.around(y)), np.uint16(np.around(r))]
                        coords.append(xyr)

            return coords


class PCTBlobDetector(pcot.ui.tabs.Tab):
    def __init__(self, node, w):
        super().__init__(w, node, "tabpctBlobDetect.ui")
        self.w.BlobDetect.clicked.connect(self.detect)
        self.w.FilterArea.toggled.connect(self.filterByArea)
        self.w.minAreaFilter.editingFinished.connect(self.areaSize)
        self.w.FilterCircularity.toggled.connect(self.filterByCircularity)
        self.w.minCircularityFilter.editingFinished.connect(self.minCircularity)
        self.w.FilterConvexity.toggled.connect(self.filterByConvexity)
        self.w.minConvexityFilter.editingFinished.connect(self.minConvexity)
        self.w.FilterInertia.toggled.connect(self.filterByInertia)
        self.w.minInertiaFilter.editingFinished.connect(self.minInertia)
        self.w.MinThreshold.editingFinished.connect(self.minThresh)
        self.w.MaxThreshold.editingFinished.connect(self.maxThresh)
        self.nodeChanged()

    def filterByArea(self):
        self.mark()
        self.node.areaFilter = self.w.FilterArea.isChecked()
        self.changed()

    def areaSize(self):
        self.mark()
        self.node.minArea = self.w.minAreaFilter.value()
        self.changed()

    def filterByCircularity(self):
        self.mark()
        self.node.circularityFilter = self.w.FilterCircularity.isChecked()
        self.changed()

    def minCircularity(self):
        self.mark()
        self.node.minCircularity = self.w.minCircularityFilter.value()
        self.changed()

    def filterByConvexity(self):
        self.mark()
        self.node.circularityFilter = self.w.FilterConvexity.isChecked()
        self.changed()

    def minConvexity(self):
        self.mark()
        self.node.minConvexity = self.w.minConvexityFilter.value()
        self.changed()

    def filterByInertia(self):
        self.mark()
        self.node.inertiaFilter = self.w.FilterInertia.isChecked()
        self.changed()

    def minInertia(self):
        self.mark()
        self.node.minInertia = self.w.minInertiaFilter.value()
        self.changed()

    def minThresh(self):
        self.mark()
        self.node.minThresh = self.w.MinThreshold.value()
        self.changed()

    def maxThresh(self):
        self.mark()
        self.node.maxThresh = self.w.MaxThreshold.value()
        self.changed()

    def detect(self):
        self.mark()
        self.node.Detect = True
        self.changed()

    def onNodeChanged(self):
        self.w.canvas.setMapping(self.node.mapping)
        self.w.canvas.setGraph(self.node.graph)
        self.w.canvas.setPersister(self.node)
        self.w.canvas.display(self.node.out)


class PCTHoughTab(pcot.ui.tabs.Tab):
    def __init__(self, node, w):
        super().__init__(w, node, "tabpctHoughTransform.ui")
        # self.w.NAME.SIGNAL.connect(self.FUNCTION WHICH CHANGES VALUE)
        self.w.KernelSize.editingFinished.connect(self.kernelChanged)
        self.w.Detect.clicked.connect(self.detectState)
        self.nodeChanged()

    def kernelChanged(self):
        self.mark()
        self.node.Kernel = self.w.KernelSize.value()
        self.changed()

    def detectState(self):
        self.mark()
        self.node.Detect = True
        self.changed()

    def onNodeChanged(self):
        self.w.canvas.setMapping(self.node.mapping)
        self.w.canvas.setGraph(self.node.graph)
        self.w.canvas.setPersister(self.node)
        self.w.canvas.display(self.node.out)
