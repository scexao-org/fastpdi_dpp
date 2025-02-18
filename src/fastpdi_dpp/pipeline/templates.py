from .config import *

__all__ = ["FASTPDI_PDI", "FASTPDI_MAXIMAL"]

DEFAULT_DIRS = {
    ProductOptions: "products",
    CalibrateOptions: "calibrated",
    FrameSelectOptions: "selected",
    RegisterOptions: "registered",
    CollapseOptions: "collapsed",
    PolarimetryOptions: "pdi",
}

FASTPDI_PDI = PipelineOptions(
    name="",
    frame_centers=dict(left=[], right=[]),
    calibrate=CalibrateOptions(
        master_dark="",
        master_flat="",
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(
        method="com", smooth=True, output_directory=DEFAULT_DIRS[RegisterOptions]
    ),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(
        output_directory=DEFAULT_DIRS[PolarimetryOptions], ip=IPOptions()
    ),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)


FASTPDI_MAXIMAL = PipelineOptions(
    name="",
    frame_centers=dict(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_dark="",
        master_flat="",
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    frame_select=FrameSelectOptions(
        cutoff=0.3, metric="normvar", output_directory=DEFAULT_DIRS[FrameSelectOptions]
    ),
    register=RegisterOptions(
        method="com", smooth=True, output_directory=DEFAULT_DIRS[RegisterOptions]
    ),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(output_directory=DEFAULT_DIRS[PolarimetryOptions]),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)
