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

import logging

from pygeoapi.process.base import (BaseProcessor, ProcessorExecuteError)

LOGGER = logging.getLogger(__name__)


# Process inputs: http://docs.ogc.org/DRAFTS/18-062.html#sc_process_inputs
# Bbox: http://docs.ogc.org/DRAFTS/18-062.html#bbox-schema

PROCESS_METADATA = {
    'version': '0.1.0',
    'id': 'landcover-prediction',
    'title': 'Landcover prediction',
    'description': 'Landcover prediction with landsat',
    'keywords': ['landcover prediction', 'landsat', 'tb-17'],
    'links': [{
        'type': 'text/html',
        'rel': 'canonical',
        'title': 'information',
        'href': 'https://github.com/geopython/pygeoapi/blob/master/pygeoapi/process/hello_world.py',
        'hreflang': 'en-US'
    }],
    'inputs': {
        'landsat-collection-id': {
            'title': 'Name',
            'description': 'Landsat coverage collection id',
            'schema': {
                'type': 'string'
            },
            'minOccurs': 1,
            'maxOccurs': 1,
            'metadata': None,  # TODO how to use?
            'keywords': ['landsat']
        },
        'bbox': {
            'title': 'Spatial bounding box',
            'description': 'Spatial bounding box in WGS84',
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
        'echo': {
            'title': 'Landcover prediction',
            'description': 'Landcover prediction with Landsat 8 Collection 2 Level 2 for water, herbs and coniferous',
            'schema': {
                'type': 'object',
                'contentMediaType': 'application/json'
            }
        }
    },
    'example': {
        "inputs": {
            "landsat-collection-id": "landsat8_c2_l2",
            "bbox": "1,2,1,2"
        }
    }
}


class LandcoverPredictionProcessor(BaseProcessor):
    """Landcover Prediction Processor"""

    def __init__(self, processor_def):
        """
        Initialize object

        :param processor_def: provider definition

        :returns: odcprovider.processes.LandcoverPredictionProcessor
        """

        super().__init__(processor_def, PROCESS_METADATA)

    def execute(self, data):

        mimetype = 'application/json'
        collection_id = data.get('landsat-collection-id', None)
        bbox = data.get('bbox', '')

        if collection_id is None:
            raise ProcessorExecuteError('Cannot process without a collection_id')
        if bbox is None:
            raise ProcessorExecuteError('Cannot process without a bbox')

        LOGGER.debug('Process inputs:\n - collection_id: {}\n - bbox: {}'.format(collection_id, bbox))
        LOGGER.debug(type(bbox))

        # Implementation steps:
        # 1) Parse process inputs
        # 2) Get array to use for the prediction with the correct bbox
        #    a) either using open data cube directly or
        #    b) making a coverage request (may be slower but enables usage of external collections)
        # 3) If necessary adapt this function https://github.com/SufianZa/Landsat-classification/blob/main/u_net.py#L208 to use, e.g., array input instead of path
        # 4) Make the prediction using this method https://github.com/SufianZa/Landsat-classification/blob/main/test.py
        # 5) Correctly encode the result of 4) as process output (geotiff)

        outputs = [{
            'id': 'echo',
            'collection_id': collection_id,
            'bbox': bbox
        }]

        return mimetype, outputs

    def __repr__(self):
        return '<LandcoverPredictionProcessor> {}'.format(self.name)
