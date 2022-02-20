# lucky imaging library functions

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.measure import centroid
from skimage.transform import SimilarityTransform, warp


def lucky_image(cube, q=0, metric="max", register="max", window=None, **kwargs):
    """
    Traditional lucky imaging

    Parameters
    ----------
    cube : np.ndarray
        Input data
    q : float, optional
        The quantile for frame selection, by default 0. Must be between 0 and 1.
    register : str, optional
        The algorithm for registering frames. "max" will align the peak pixel to the center, "com" will align the center of mass, and "dft" uses phase cross-correlation for registering frames. By default "max".
    metric : str, optional
        The metric for which frame selection will be applied. The available metrics are "max" and "min". By default "max".
    window : int, optional
        If not None, will only measure the frame selection metric in a centered box with this width, by default None.
    **kwargs
        additional keyword arguments will be passed to the registration method (e.g. `upsample_factor`)
    """

    if q < 0 or q >= 1:
        raise ValueError(
            "The frame selection quantile must be less than or equal to 0 (no discard) and less than 1"
        )

    tmp_cube = cube.copy()
    # do frame selection
    if q > 0:
        values = measure_metric(cube, metric)
        cut = np.quantile(values, q)
        tmp_cube = cube[values >= cut]

    if register == "dft":
        refidx = measure_metric(cube, metric).argmax()
        refframe = cube[refidx]

    out = np.zeros(cube.shape[1:], "f4")
    center = image_center(tmp_cube)[1:]
    for i in range(tmp_cube.shape[0]):
        frame = tmp_cube[i]
        # measure offset
        if register == "max":
            idx = np.argmax(frame)
            delta = center - idx
        elif register == "com":
            idx = centroid(frame)
            delta = center - idx
        elif register == "dft":
            delta = phase_cross_correlation(
                refframe, frame, return_error=False, **kwargs
            )

        tform = SimilarityTransform(translation=delta)
        shifted = warp(frame, tform)
        out += shifted

    return out


def measure_metric(cube, metric):
    if metric == "max":
        values = np.max(cube, axis=(1, 2))
    elif metric == "min":
        values = np.min(cube, axis=(1, 2))
    else:
        raise ValueError(
            f"Did not recognize frame selection metric {metric}. Please choose between 'max' and 'min'."
        )

    return values


def image_center(array):
    return np.asarray(array.shape) / 2
