import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
import numpy as np
import cv2 as cv
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from skimage.exposure import equalize_hist
from os import walk
import re

from config import LAND_COVER_FILE, SUPPORTED_BANDS, REFLECTANCE_MAX_BAND, PADDING_EDGE


def merge_reprojected_bands(datasets_folder):
    datasets_dict = dict()
    i = 0
    for root, dirs, files in walk(datasets_folder):
        f = []
        for file in files:
            if re.search('SR_B[2-7].TIF$', file) is not None:
                f.append(Path(root, file))
        if len(f) > 0:
            datasets_dict['dataset%d' % i] = str(Path(root, Path(file).stem[:-1]))
            i += 1

    ds = rasterio.open(LAND_COVER_FILE)
    for dataset, files in datasets_dict.items():
        # get dimensions and transformation form the first band in the dataset
        with rasterio.open(Path(files + '2').with_suffix('.TIF'), read='r+') as band:
            transform, width, height = calculate_default_transform(
                band.crs, ds.crs, band.width, band.height, *band.bounds, resolution=ds.res)
            kwargs = band.meta.copy()
            kwargs.update({
                'crs': ds.crs,
                'transform': transform,
                'width': width,
                'height': height,
                'count': 7
            })
            with rasterio.open(Path(datasets_folder, '%s.tif' % dataset), 'w', **kwargs) as dst:
                print('Reprojecting bands of %s' % dataset)
                for i, b in enumerate(SUPPORTED_BANDS, start=1):
                    with rasterio.open(Path(files + '%d' % b).with_suffix('.TIF')) as band:
                        reproject(
                            source=rasterio.band(band, 1),
                            destination=rasterio.band(dst, b),
                            src_transform=band.transform,
                            src_crs=band.crs,
                            dst_transform=transform,
                            dst_crs=ds.crs,
                            resampling=Resampling.nearest)
    return list(datasets_dict.keys())


def rotate_datasets(landsat_dataset_path, enhance_colors=False, show_preprocessing_steps=False, label=True):
    with rasterio.open(landsat_dataset_path) as l_sat:
        west, south, east, north = l_sat.bounds
        bands = []
        masks = []
        # collect and normalize spectral bands
        for band_num in SUPPORTED_BANDS:
            band = l_sat.read(band_num)
            if band_num in [2, 3, 4]:
                masks.append(band != 0)
            band = band / REFLECTANCE_MAX_BAND
            bands.append(band)

        # stacking Multi-spectral image containing -> (Blue, Green, Red, NIR, SWIR 1, SWIR 2)
        ls_original = np.array(bands).transpose([1, 2, 0])

        # extract mask from the bands
        mask = np.mean(np.array(masks).transpose([1, 2, 0]), axis=2)
        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        # calculate the angle to perform the affine transformation of the (rotated) dataset
        coords = np.column_stack(np.where(mask))
        angle = cv.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        w, h, _ = ls_original.shape
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        # perform affine transformation of landsat
        ls_rotated = cv.warpAffine(ls_original, M, (h, w),
                                   flags=cv.INTER_NEAREST,
                                   borderMode=cv.BORDER_CONSTANT)

        # perform affine transformation the original mask
        mask = cv.warpAffine(mask, M, (h, w),
                             flags=cv.INTER_NEAREST,
                             borderMode=cv.BORDER_CONSTANT)
        # crop
        x, y = np.nonzero(mask)
        ls_cropped = ls_rotated[np.ix_(np.unique(x), np.unique(y))]

        # remove padding on the edges
        ls_cropped = ls_cropped[PADDING_EDGE:-PADDING_EDGE, PADDING_EDGE:-PADDING_EDGE]

        if not label:
            return ls_cropped

        with rasterio.open(LAND_COVER_FILE) as ds:
            # reading a window oo landcover dataset according to landsat boundries
            lc_original = ds.read(1, window=from_bounds(west, south, east, north, transform=ds.transform))
            # perform affine transformation of landcover
            lc_rotated = cv.warpAffine(lc_original, M, (h, w),
                                       flags=cv.INTER_NEAREST,
                                       borderMode=cv.BORDER_CONSTANT)

            # masking land cover dataset
            lc_masked = lc_rotated * mask

            # crop
            lc_cropped = lc_masked[np.ix_(np.unique(x), np.unique(y))]

            # remove padding on the edges
            lc_cropped = lc_cropped[PADDING_EDGE:-PADDING_EDGE, PADDING_EDGE:-PADDING_EDGE]

            # enhance colors
            if enhance_colors:
                ls_cropped[:, :, 0] = equalize_hist(ls_cropped[:, :, 0])
                ls_cropped[:, :, 1] = equalize_hist(ls_cropped[:, :, 1])
                ls_cropped[:, :, 2] = equalize_hist(ls_cropped[:, :, 2])

            # show steps
            if show_preprocessing_steps:
                fig, ax = plt.subplots(nrows=2, ncols=4)
                ax[0][0].imshow(ls_original)
                ax[1][0].imshow(lc_original, cmap='nipy_spectral')
                ax[0][0].title.set_text('original')
                ax[0][0].set_axis_off()
                ax[1][0].set_axis_off()

                ax[0][1].imshow(ls_rotated)
                ax[1][1].imshow(lc_rotated, cmap='nipy_spectral')
                ax[0][1].title.set_text('rotated')
                ax[0][1].set_axis_off()
                ax[1][1].set_axis_off()

                ax[0][2].imshow(ls_rotated)
                ax[1][2].imshow(lc_masked, cmap='nipy_spectral')
                ax[0][2].title.set_text('masked')
                ax[0][2].set_axis_off()
                ax[1][2].set_axis_off()

                ax[0][3].imshow(ls_cropped)
                ax[1][3].imshow(lc_cropped, cmap='nipy_spectral')
                ax[0][3].title.set_text('cropped ')
                ax[0][3].set_axis_off()
                ax[1][3].set_axis_off()
                plt.show()
            return ls_cropped, lc_cropped


def getMultiSpectral(landsat_dataset_path):
    with rasterio.open(landsat_dataset_path) as l_sat:
        bands = []
        masks = []
        metadata = l_sat.meta.copy()
        metadata.update({'count': 1})
        # collect and normalize spectral bands
        for band_num in SUPPORTED_BANDS:
            band = l_sat.read(band_num)
            if band_num in [2, 3, 4]:
                masks.append(band != 0)
            band = band / REFLECTANCE_MAX_BAND
            bands.append(band)

        # stacking Multi-spectral image containing -> (Blue, Green, Red, NIR, SWIR 1, SWIR 2)
        ls_original = np.array(bands).transpose([1, 2, 0])

        # extract mask from the bands
        mask = np.mean(np.array(masks).transpose([1, 2, 0]), axis=2)
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
    return ls_original, mask, metadata
