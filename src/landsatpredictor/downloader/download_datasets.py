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
import os
from pathlib import Path

import requests
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
from tqdm import tqdm

# download land cover dataset
land_cover_url = 'https://opendata.nfis.org/downloads/forest_change/CA_forest_VLCE_2015.zip'
response = requests.get(land_cover_url, stream=True)
Path('./landcover').mkdir(parents=True, exist_ok=True)
with open(str(Path('./landcover', 'land_cover.zip')), "wb") as handle:
    for data in tqdm(response.iter_content()):
        handle.write(data)

# download landsat dataset
username, password = os.getenv('UN_EarthE'), os.getenv('PW_EarthE')
api = API(username, password)

# search area
west, south, east, north = -127.4439, 48.9225, -84.1984, 58.9953

# search time range
start_date = '2015-07-15'
end_date = '2015-08-30'

# Search for Landsat 8 https://pypi.org/project/landsatxplore/
scenes = api.search(
    dataset='landsat_ot_c2_l2',
    bbox=(west, south, east, north),
    start_date=start_date,
    end_date=end_date,
    max_cloud_cover=1
)

print(f"{len(scenes)} scenes found.")
api.logout()

ee = EarthExplorer(username, password)

for scene in scenes:
    print(scene['entity_id'])
    try:
        ee.download(identifier=scene['entity_id'], dataset='landsat_ot_c2_l2', output_dir='../train')
    except Exception as e:
        print(e)
ee.logout()
