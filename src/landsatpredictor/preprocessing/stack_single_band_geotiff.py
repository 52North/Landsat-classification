from pathlib import Path

import rasterio

path = "../../data/LC08_L2SP_035024_20150813_20200909_02_T1"
multi_image = [rasterio.open(band_path) for band_path in sorted(list(Path(path).glob('*SR_B[2-7].TIF')))]

meta = multi_image[0].meta.copy()
meta.update(count=7)
with rasterio.open(Path(path, 'LC08_L2SP_035024_20150813_20200909_02_T1_merged_1-7.tif'), 'w', **meta) as dst:
    for i, band in enumerate(multi_image, start=2):
        dst.write(band.read(1), i)
        band.close()
