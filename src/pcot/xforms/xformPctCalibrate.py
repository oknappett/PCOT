from pcot.datum import Datum
from pcot.xform import xformtype, XFormType



@xformtype
class XformPctCalibrate(XFormType):
    """Locates the pct and returns the centre pixel coordinate
    of each target circle"""

    def __init__(self):
        super().__init__("pctCalibrate", "calibration", "0.0.0")
        self.addInputConnector("img", Datum.IMG)
        self.addOutputConnector("", Datum.IMG)

    def createTab(self, n, w):
        return TabImage(n, w)

    def init(self, node):
        node.out = None

    def perform(self, node):
        data = node.getInput(0, Datum.IMG)
        if data is not None:
            img = data.copy()
            img = img.img[:,:,1]