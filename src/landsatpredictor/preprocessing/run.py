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
from pathlib import Path

from image_registration import merge_reprojected_bands, rotate_datasets
from patches_generator import generate_patches
from ..config import selected_classes, TRAIN_DATASETS, TEST_DATASETS

# reproject each dataset then obtain a list of paths
reprojected_train_datasets = merge_reprojected_bands(TRAIN_DATASETS)
reprojected_test_datasets = merge_reprojected_bands(TEST_DATASETS)

# create patches from the training datasets
for i, dataset in enumerate(reprojected_train_datasets):
    generate_patches(*rotate_datasets(Path(TRAIN_DATASETS, '%s.tif' % dataset)), train_flag=True, data_id=i.__str__(),
                     class_assignment=selected_classes, patches_per_map=1300)

# create patches from the testing datasets
for i, dataset in enumerate(reprojected_test_datasets):
    generate_patches(*rotate_datasets(Path(TEST_DATASETS, '%s.tif' % dataset)), train_flag=False, data_id=i.__str__(),
                     class_assignment=selected_classes, patches_per_map=200)
