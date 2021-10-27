# =================================================================
# Copyright (C) 2021-2021 52°North Spatial Information Research GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================
from __future__ import annotations

import json
import logging
import os

import time
from typing import Tuple, Any
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path
from urllib.error import HTTPError

import requests
from .u_net import UNET
from .preprocessing.image_registration import get_multi_spectral
from pygeoapi.process.base import (BaseProcessor, ProcessorExecuteError)

BASE_URL = "https://17.testbed.dev.52north.org/geodatacube/collections/{}/coverage?f=NetCDF&bbox={}"

LOGGER = logging.getLogger(__name__)

#
# LINKS
#
# Process inputs
#   https://github.com/opengeospatial/ogcapi-processes/blob/master/core/examples/json/ProcessDescription.json#L14
#   http://docs.ogc.org/DRAFTS/18-062.html#sc_process_inputs
#   Bbox:
#   http://docs.ogc.org/DRAFTS/18-062.html#bbox-schema
#   https://github.com/opengeospatial/ogcapi-coverages#query-parameters-optional-conformance-classes
#
# Process outputs
#   https://github.com/opengeospatial/ogcapi-processes/blob/master/core/examples/json/ProcessDescription.json#L199
#   Image
#   https://github.com/opengeospatial/ogcapi-processes/blob/master/core/examples/json/ProcessDescription.json#L318-L325
#
# Implementation
# Async processing pygeoapi:
#       https://docs.pygeoapi.io/en/latest/data-publishing/ogcapi-processes.html#asynchronous-support
#
PROCESS_METADATA = {
    'version': '0.1.0',
    'id': 'landcover-prediction',
    'title': 'Land cover prediction',
    'description': 'Land cover prediction with Landsat 8',
    'keywords': ['land cover prediction', 'landsat 8', 'tb-17'],
    'jobControlOptions': 'async-execute',
    'outputTransmission': ['value'],
    'links': [
        {
            'type': 'text/html',
            'rel': 'canonical',
            'title': 'Processor Repository',
            'href': 'https://github.com/52North/Landsat-classification/blob/main/README.md',
            'hreflang': 'en-US'
        },
        {
            'type': 'text/html',
            'rel': 'canonical',
            'title': 'Landsat 8 Collection 2 Level 2',
            'href': 'https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-level-2-science-products',
            'hreflang': 'en-US'
        }
    ],
    'inputs': {
        'collection': {
            'title': 'Coverage',
            'description': 'url of the OGC API Coverages collection providing the Landsat 8 Collection 2 '
                           'Level 2 data (must start with http or https and include the following bands:'
                           ' blue, green, red, nir, swir1, swir2)',
            'schema': {
                'type': 'string'
            },
            'minOccurs': 1,
            'maxOccurs': 1,
            # TODO how to use?
            'metadata': None,
            'keywords': ['landsat']
        },
        'bbox': {
            'title': 'Spatial bounding box',
            'description': 'Spatial bounding box in WGS84 (format: "min lon, min lat, max lon, max lat")',
            'schema': {
                'type': 'string'
            },
            'minOccurs': 1,
            'maxOccurs': 1,
            'metadata': None,
            'keywords': ['bbox']
        }
    },
    'outputs': {
        'prediction': {
            'title': 'Land cover prediction',
            'description':
                'Land cover prediction with Landsat 8 Collection 2 Level 2 for no change (=1), '
                'water (=2), coniferous (=3) and herbs (=4) (no data=0)',
            'schema': {
                'type': 'string',
                'format': 'byte',
                'contentMediaType': 'image/tiff; application=geotiff'
            }
        }
    },
    'example': {
        'inputs': {
            'collection': 'https://17.testbed.dev.52north.org/geodatacube/collections/landsat8_c2_l2',
            'bbox': '-104.7,51.4,-103.0,52.6'
        },
         # pygeoapi uses mode: async
        'jobControlOptions': ['async-execute'],
        'outputTransmission': ['value'],
        'response': 'raw'
    }
}


class ModelCache:
    """
    Stores not changing trained model to be used by each instance of the LandcoverPredictionProcessor.
    Implementation follows:

        https://python-patterns.guide/gang-of-four/singleton/
    """

    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls) -> ModelCache:
        LOGGER.debug("instance() called")
        #
        #   Init model and its data as global pickled singleton
        #
        if cls._instance is None:
            LOGGER.debug("Creating instance of class '{}'...".format(UNET))
            cls._instance = UNET()
            LOGGER.debug('...DONE.')
        else:
            LOGGER.debug("Instance of class '{}' already existing".format(UNET))
        return cls._instance


class LandcoverPredictionProcessor(BaseProcessor):
    """Landcover Prediction Processor"""

    def __init__(self, processor_def):
        """
        Initialize object

        :param processor_def: provider definition

        :returns: odcprovider.processes.LandcoverPredictionProcessor
        """

        super().__init__(processor_def, PROCESS_METADATA)
        self.model = ModelCache.instance()

    def execute(self, data: dict) -> Tuple[str, Any]:

        bbox, collection = self._parse_inputs(data)
        input_landsat_bands_normalized, visual_light_reflectance_mask, metadata = self._process_collection(collection, bbox)

        LOGGER.debug('Requesting prediction for "{}"'.format(collection))
        result_file_path = self.model.estimate_raw_landsat(input_landsat_bands_normalized, visual_light_reflectance_mask, metadata, trim=20)
        LOGGER.debug('Prediction received. Result in "{}"'.format(result_file_path))

        # ToDo: Correctly return process output as geotiff
        mimetype = 'image/tiff; application=geotiff'
        with open(result_file_path, 'r+b') as file:
            return mimetype, file.read()

    def _parse_inputs(self, data):
        LOGGER.debug("RAW Inputs:\n{}".format(json.dumps(data, indent=4)))
        # 1) Parse process inputs
        collection = data.get('collection', None)
        bbox = data.get('bbox', None)
        if collection is None:
            raise ProcessorExecuteError('Cannot process without a collection')
        if bbox is None:
            raise ProcessorExecuteError('Cannot process without a bbox')
        LOGGER.debug('Parsed Process inputs')
        LOGGER.debug('collection       : {}'.format(collection))
        LOGGER.debug('bbox             : {}'.format(bbox))
        bbox_coords = [s.strip() for s in bbox.split(",")]
        if len(bbox_coords) != 4:
            raise ProcessorExecuteError("Received bbox '{}' could not be split into four (4) elements by ','."
                                        .format(bbox))
        bbox_float_coords = list(map(float, bbox_coords))
        if not all(isinstance(x, float) for x in bbox_float_coords):
            raise ProcessorExecuteError("Received bbox '{}' could not be converted completely to integer."
                                        .format(bbox))
        return bbox, collection

    def _process_collection(self, collection, bbox):
        if collection.startswith('http'):
            if not collection.endswith('/'):
                collection = collection + '/'
            coverage_download_url = collection + 'coverage?f=GeoTIFF&bbox=' + bbox

            LOGGER.debug("Requesting coverage from '{}'".format(coverage_download_url))
            try:
                with requests.get(coverage_download_url, verify=False, stream=True) as request:
                    request.raise_for_status()

                    with MemoryFile(request.content) as memfile:
                        with memfile.open() as dataset:
                            return get_multi_spectral(dataset)

                    # ToDo use correct temp file and use tmp file name as input for unet.estimate_raw
                    # with open('/tmp/temp.geotiff', 'wb') as file:
                    #     for chunk in request.iter_content(chunk_size=8192):
                    #         file.write(chunk)
            except HTTPError as err:
                msg = 'Requesting input data failed: {}'.format(coverage_download_url)
                LOGGER.error(msg)
                raise ProcessorExecuteError(msg)
            # write response to temporary file used as input for prediction/estimation function
            # ToDo: add logger output, e.g. error/warning if request wasn't successful

        elif collection.startswith('file'):
            landsat_file_path = str(Path(collection).resolve())

            with rasterio.open(landsat_file_path) as dataset:
                return get_multi_spectral(dataset)
        else:
            raise(ProcessorExecuteError("Invalid collection input received: '{}'.".format(collection)))

    def __repr__(self):
        return '<LandcoverPredictionProcessor> {}'.format(self.name)
