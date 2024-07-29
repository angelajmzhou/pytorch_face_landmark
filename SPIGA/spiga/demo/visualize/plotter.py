# Demo libs
from .layouts import plot_basics as pl_basic
from .layouts import plot_bbox as pl_bbox
from .layouts import plot_landmarks as pl_lnd
from .layouts import plot_headpose as pl_hpose


class Plotter:

    def __init__(self):
        self.basic = pl_basic.BasicLayout()
        self.bbox = pl_bbox.BboxLayout()
        self.landmarks = pl_lnd.LandmarkLayout()
        self.hpose = pl_hpose.HeadposeLayout()
