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

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

original_classes = dict(no_change=0,
                        water=20,
                        snow_ice=31,
                        rock_rubble=32,
                        exposed_barren_land=33,
                        bryoids=40,
                        shrubland=50,
                        wetland=80,
                        wetlandtreed=81,
                        herbs=100,
                        coniferous=210,
                        broadleaf=220,
                        mixedwood=230)
classes_names = list(original_classes.keys())
model_classes = {c: idx for idx, c in enumerate(classes_names)}


def generate_patches(train_image, label_image, train_flag=True, bands=None,
                     class_assignment=[], data_id='',
                     patch_size=256,
                     patches_per_map=15):
    # get all classes if no specific were given
    if len(class_assignment) == 0:
        class_assignment = classes_names

    # calculate random indices to get patches
    sampling_weights = label_image[patch_size // 2:-patch_size // 2, patch_size // 2:-patch_size // 2].astype(np.float_)
    linear = np.cumsum(sampling_weights)
    linear /= linear[-1]
    indices = np.searchsorted(linear, np.random.random_sample(patches_per_map), side='right')

    if train_flag:
        # splitting the image into patches 80% train, 20% validation
        train, validation= train_test_split(indices, test_size=0.20, shuffle=True)
        data = dict(train=train, validation=validation)
    else:
        # all dataset is used for testing
        data = dict(test=indices)

    for subset_name, subset in data.items():
        for i, idx in enumerate(subset):
            x = idx % sampling_weights.shape[1]
            y = idx // sampling_weights.shape[1]
            input_patch = train_image[y:y + patch_size, x:x + patch_size, :]
            label_patch = label_image[y:y + patch_size, x:x + patch_size]
            label_patch_converted = np.zeros_like(label_patch)

            # create categorical mask
            for index, c in enumerate(class_assignment, start=1):
                label_patch_converted[label_patch == original_classes[c]] = index
            ######
            # TODO consider of having pixel-wise balance across the dataset
            ######
            # create directories
            Path('./dataset', subset_name, 'RGBinputs', 'input').mkdir(parents=True, exist_ok=True)
            Path('./dataset', subset_name, 'NIRinputs', 'input').mkdir(parents=True, exist_ok=True)
            Path('./dataset', subset_name, 'labels', 'label').mkdir(parents=True, exist_ok=True)

            # save patches
            # RGB
            Image.fromarray((input_patch[:, :, :3] * 255).astype(np.uint8)).save(
                os.path.join('dataset/{0}/{1}/{2}'.format(subset_name, 'RGBinputs', 'input'),
                             "{}img-{}.tiff".format(data_id, i)))
            # NIR
            Image.fromarray((input_patch[:, :, 3:] * 255).astype(np.uint8)).save(
                os.path.join('dataset/{0}/{1}/{2}'.format(subset_name, 'NIRinputs', 'input'),
                             "{}img-{}.tiff".format(data_id, i)))
            # Labels
            Image.fromarray(label_patch_converted.astype(np.uint8)).save(
                os.path.join('dataset/{0}/{1}/{2}'.format(subset_name, 'labels', 'label'),
                             "{}img-{}.tiff".format(data_id, i)))
