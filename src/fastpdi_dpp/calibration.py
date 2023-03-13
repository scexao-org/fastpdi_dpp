# library functions for common calibration tasks like
# dark subtraction, collapsing cubes
import multiprocessing as mp
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple
from numpy.typing import NDArray

import astropy.units as u
import cv2
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from tqdm.auto import tqdm
from astroscrappy import detect_cosmics

from fastpdi_dpp.constants import DEFAULT_NPROC, SUBARU_LOC, PA_OFFSET
from fastpdi_dpp.headers import fix_header, parallactic_angle
from fastpdi_dpp.indexing import frame_center
from fastpdi_dpp.image_processing import (
    collapse_cube,
    collapse_frames_files,
)
from fastpdi_dpp.util import get_paths, wrap_angle
from fastpdi_dpp.wcs import apply_wcs, get_coord_header

__all__ = [
    "calibrate_file",
    "make_dark_file",
    "make_flat_file",
    "make_master_dark",
    "make_master_flat",
]


def filter_empty_frames(cube):
    finite_mask = np.isfinite(cube)
    nonzero_mask = cube != 0
    combined = finite_mask & nonzero_mask
    inds = np.any(combined, axis=(1, 2))
    return cube[inds]


def calibrate_file(
    filename: str,
    hdu: int = 1,
    dark_filename: Optional[str] = None,
    flat_filename: Optional[str] = None,
    force: bool = False,
    bpfix: bool = False,
    coord: Optional[SkyCoord] = None,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="calib", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    header = fits.getheader(path, ext=hdu)
    # if wollaston is in, assume PDI mode
    if header["X_IRCWOL"] == "IN":
        outpath_left = outpath.with_stem(f"{outpath.stem}_left")
        outpath_right = outpath.with_stem(f"{outpath.stem}_right")
        if not force and outpath_left.is_file() and outpath_right.is_file():
            return outpath_left, outpath_right

    cube = fits.getdata(path, ext=hdu)
    cube = filter_empty_frames(cube)
    cube[:, :2] = 0
    # fix header
    header = fix_header(header)
    time = Time(header["MJD"], format="mjd", scale="ut1", location=SUBARU_LOC)
    if coord is None:
        coord_now = get_coord_header(header, time)
    else:
        coord_now = coord.apply_space_motion(time)

    header["RA"] = coord_now.ra.to_string(unit=u.hourangle, sep=":")
    header["DEC"] = coord_now.dec.to_string(unit=u.deg, sep=":")
    parang = parallactic_angle(time, coord_now)
    header["PARANG"] = parang, "[deg] derotation angle for North up"
    header["PA"] = wrap_angle(parang - PA_OFFSET), "[deg] parallactic angle of target"
    header = apply_wcs(header, parang=parang)

    # dark correction
    if dark_filename is not None:
        dark_path = Path(dark_filename)
        dark = fits.getdata(dark_path)
        cube -= dark
    # flat correction
    if flat_filename is not None:
        flat_path = Path(flat_filename)
        flat = fits.getdata(flat_path)
        flat[flat < 0.1] = 1e4
        cube /= flat
    # bad pixel correction
    if bpfix:
        mean_frame = np.mean(cube, axis=0)
        mask, _ = fix_bad_pixels(mean_frame, header)
        for i in range(cube.shape[0]):
            low_pass = cv2.medianBlur(cube[i], 3)
            cube[i, mask] = low_pass[mask]
    # flip data
    cube = np.flip(cube, axis=-2)
    header = apply_wcs(header)
    # deinterleave
    if header["X_IRCWOL"] == "IN":
        midx = cube.shape[-1] // 2
        left = cube.copy()
        left[..., midx:] = np.nan
        header["BEAM"] = "left"
        fits.writeto(outpath_left, left, header, overwrite=True)

        right = cube
        right[..., :midx] = np.nan
        header["BEAM"] = "right"
        fits.writeto(outpath_right, right, header, overwrite=True)
        return outpath_left, outpath_right

    fits.writeto(outpath, cube, header=header, overwrite=True)
    return outpath


def fix_bad_pixels(frame, header, pad_width=3, **kwargs):
    padded = np.pad(frame, pad_width=pad_width, mode="reflect")
    # expected noise frome read noise and Poisson dark current
    read_noise = 0.17**2
    dark_noise = 7.5 * header["DET-NSMP"] * header["EXPTIME"] * header["DETGAIN"]
    expected_noise = np.sqrt(read_noise**2 + dark_noise)
    default_kwargs = {
        "niter": 5,
        "sepmed": False,
        "objlim": 2,
        "gain": 0.48,
        "readnoise": expected_noise,
    }
    default_kwargs.update(**kwargs)
    mask, clean_frame = detect_cosmics(padded, **kwargs)
    # depad
    mask = mask[pad_width:-pad_width, pad_width:-pad_width]
    clean_frame = clean_frame[pad_width:-pad_width, pad_width:-pad_width]
    return mask, clean_frame


def make_dark_file(filename: str, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    cube, header = fits.getdata(
        path,
        ext=1,
        header=True,
    )
    cube = filter_empty_frames(cube)
    cube[:, :2] = 0
    master_dark, header = collapse_cube(cube, header=header, **kwargs)
    mask, clean_dark = fix_bad_pixels(master_dark, header)

    fits.writeto(
        outpath,
        clean_dark,
        header=header,
        overwrite=True,
    )
    return outpath


def make_flat_file(filename: str, force=False, dark_filename=None, **kwargs):
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    cube, header = fits.getdata(
        path,
        ext=1,
        header=True,
    )
    cube = filter_empty_frames(cube)
    if dark_filename is not None:
        dark_path = Path(dark_filename)
        header["MDARK"] = (dark_path.name, "file used for dark subtraction")
        master_dark = fits.getdata(
            dark_path,
        )
        cube = cube - master_dark
    cube[:, :2] = 0
    master_flat, header = collapse_cube(cube, header=header, **kwargs)
    mask, clean_flat = fix_bad_pixels(master_flat, header)
    # normalize flat field
    clean_flat /= np.nanmedian(clean_flat)

    fits.writeto(
        outpath,
        clean_flat,
        header=header,
        overwrite=True,
    )
    return outpath


def sort_calib_files(filenames: list[PathLike]) -> dict[Tuple, Path]:
    darks_dict = {}
    for filename in filenames:
        path = Path(filename)
        ext = 1 if ".fits.fz" in path.name else 0
        header = fits.getheader(path, ext=ext)
        key = (header["DETGAIN"], header["EXPTIME"] * header["DET-NSMP"])
        if key in darks_dict:
            darks_dict[key].append(path)
        else:
            darks_dict[key] = [path]
    return darks_dict


def make_master_dark(
    filenames: list[PathLike],
    collapse: str = "median",
    name: str = "master_dark",
    force: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> list[Path]:
    # prepare input filenames
    file_inputs = sort_calib_files(filenames)
    # make darks for each camera
    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd()

    # get names for master darks and remove
    # files from queue if they exist
    outnames = {}
    with mp.Pool(num_proc) as pool:
        jobs = []
        for key, filelist in file_inputs.items():
            gain, exptime = key
            outname = outdir / f"{name}_em{gain:.0f}_{exptime*1e3:05.0f}ms.fits"
            outnames[key] = outname
            if not force and outname.is_file():
                continue
            # collapse the files required for each dark
            for path in filelist:
                kwds = dict(
                    output_directory=path.parent.parent / "collapsed",
                    force=force,
                    method=collapse,
                )
                jobs.append(pool.apply_async(make_dark_file, args=(path,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Collapsing dark frames")
        frames = [job.get() for job in iter]
        # create master frames from collapsed files
        collapsed_inputs = sort_calib_files(frames)
        jobs = []
        for key, filelist in collapsed_inputs.items():
            kwds = dict(
                output=outnames[key],
                method=collapse,
                force=force,
            )
            jobs.append(pool.apply_async(collapse_frames_files, args=(filelist,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Making master darks")
        [job.get() for job in iter]

    return list(outnames.values())


def make_master_flat(
    filenames: list[PathLike],
    master_darks: Optional[list[PathLike]] = None,
    collapse: str = "median",
    name: str = "master_flat",
    force: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> list[Path]:
    # prepare input filenames
    file_inputs = sort_calib_files(filenames)
    master_dark_inputs = {key: None for key in file_inputs.keys()}
    if master_darks is not None:
        inputs = sort_calib_files(master_darks)
        for key in file_inputs.keys():
            if key in inputs:
                master_dark_inputs[key] = inputs[key][0]
    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd()

    # get names for master darks and remove
    # files from queue if they exist
    outnames = {}
    with mp.Pool(num_proc) as pool:
        jobs = []
        for key, filelist in file_inputs.items():
            gain, exptime = key
            outname = outdir / f"{name}_em{gain:.0f}_{exptime*1e3:05.0f}ms.fits"
            outnames[key] = outname
            if not force and outname.is_file():
                continue
            # collapse the files required for each flat
            for path in filelist:
                kwds = dict(
                    output_directory=path.parent.parent / "collapsed",
                    dark_filename=master_dark_inputs[key][0],
                    force=force,
                    method=collapse,
                )
                jobs.append(pool.apply_async(make_flat_file, args=(path,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Collapsing flat frames")
        frames = [job.get() for job in iter]
        # create master frames from collapsed files
        collapsed_inputs = sort_calib_files(frames)
        jobs = []
        for key, filelist in collapsed_inputs.items():
            kwds = dict(
                output=outnames[key],
                method=collapse,
                force=force,
            )
            jobs.append(pool.apply_async(collapse_frames_files, args=(filelist,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Making master flats")
        [job.get() for job in iter]

    return outnames
