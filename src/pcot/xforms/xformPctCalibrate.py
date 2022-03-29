
import cv2 as cv
import numpy as np

import pcot
from pcot.datum import Datum
from pcot.xform import xformtype, XFormType


class XformPctCalibrate(XFormType):
    """Locates the pct and returns the centre pixel coordinate
    of each target circle"""

    def __init__(self, name, group, version):
        super().__init__(name, group, version)
        # single input image
        self.addInputConnector("input", Datum.IMG)
        # single output, it's an image for now but will need to decide on a
        # output type at a later date -> the centre pixels of the targets
        self.addOutputConnector("output", Datum.IMG)

    def createTab(self, n, w):
        pass

    def init(self, node):
        # output set to null initially
        node.out = None
        node.Kernel = 11

    @staticmethod
    def thresh(self, img):
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

    @staticmethod
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

    def Cluster(self, Circles):
        if Circles is None:
            return None
        else:
            # loop through all circles
            AverageCircles = []
            copy = Circles.copy()
            clustered = []
            for c in copy:
                x, y, r = c[0], c[1], c[2]
                SameCircles = []
                for j in copy:
                    # loop through all circles and compare to current circle
                    if j is not c:
                        Jx, Jy, Jr = j[0], j[1], j[2]
                        # if current circle centre is inside other circle then remove circle
                        if ((Jx <= (x + r)) and (Jx >= (x - r))) and ((Jy <= (y + r)) and (Jy >= (y - r))):
                            copy.remove(j)
                            SameCircles.append([Jx, Jy, Jr])

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

                copy.remove(c)

            print("cluster: ", clustered)
            return clustered

    def detect(self):
        pass

    def perform(self, node):
        print(node.Kernel)
        data = node.getInput(0, Datum.IMG)
        if data is None:
            out = None
        else:
            img = data.copy()
            k = node.Kernel
            out = self.detect(img, k)
            edges = self.edge(img, k)
            out = out
        node.out = Datum(Datum.IMG, out)
        node.setOutput(0, node.out)


@xformtype
class HoughCircles(XformPctCalibrate):
    """Class for Hough Circle detection"""

    def __init__(self):
        super().__init__("Hough Circles", "calibration", "0.0.0")
        # single input image
        self.addInputConnector("", Datum.IMG)
        # single output, it's an image for now but will need to decide on a
        # output type at a later date -> the centre pixels of the targets
        self.addOutputConnector("", Datum.IMG)

    def createTab(self, n, w):
        # standard tab for now, just outputs the image after the
        # xform is applied within the node
        return PCTCalibTab(n, w)

    def detect(self, img, k):
        if img is None:
            return None
        else:
            copy = img.copy()
            circleAmount = []
            rgb = copy.rgbImage()
            # edge detector on image
            edges = self.edge(copy, k)
            circleArray = []
            for i in range(edges.channels):
                c = edges.img[:, :, i]
                # convert edge image to a np array of uint
                cArray = np.array(c).astype(np.uint8)
                # hough circle detection
                circles = cv.HoughCircles(cArray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0,
                                          maxRadius=0)
                if circles is not None:
                    circleNo = 0
                    circles = np.uint16(np.around(circles))
                    # draw circles on rgb image
                    for j in circles[0, :]:
                        circleNo += 1
                        x, y, r = j[0], j[1], j[2]
                        circleArray.append([x, y, r])
                        # cv.circle(rgb.img, (x, y), r, (1, 0, 0), 1)
                        # cv.circle(rgb.img, (x, y), 1, (0, 1, 0), 1)
                    circleAmount.append(circleNo)
            ClusteredCircles = self.Cluster(circleArray)
            for n in ClusteredCircles:
                x, y, r = n[0], n[1], n[2]
                cv.circle(rgb.img, (x, y), r, (1, 0, 0), 1)
                cv.circle(rgb.img, (x, y), 1, (0, 1, 0), 1)
            print(circleAmount, np.mean(circleAmount))
            # return rgb image with circles drawn on
            return rgb


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


