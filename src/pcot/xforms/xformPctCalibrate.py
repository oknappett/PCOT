from pcot.datum import Datum
from pcot.xform import xformtype, XFormType



@xformtype
class XformPctCalibrate(XFormType):
    """Locates the pct and returns the centre pixel coordinate
    of each target circle"""

    def __init__(self):
        super().__init__("pctCalibrate", "calibration", "0.0.0")
        self.addInputConnector("", Datum.IMG)
        self.addOutputConnector("", Datum.IMG)

