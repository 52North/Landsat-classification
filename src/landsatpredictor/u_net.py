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
import pickle
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Conv2DTranspose, Dropout
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from .config import selected_classes, colors, colors_legend


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def get_tiles_generator(w, h, window_size, trim, in_image):
    step_size = window_size - trim * 2
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            x_overflow = x + window_size
            y_overflow = y + window_size
            if x_overflow > w and y_overflow > h:
                x_overflow = w - 1
                y_overflow = h - 1
            elif x_overflow > w:
                x_overflow = w - 1
            elif y_overflow > h:
                y_overflow = h - 1
            yield in_image[:, x:x_overflow, y:y_overflow, :], x, y, x_overflow, y_overflow


class UNET:
    def __init__(self, weight_file: str = str(Path('data/3_class_best_weight.hdf5')), batch_size: int = 64,
                 epochs: int = 30, window_size: int = 256):
        self.bands = 6
        self.batch_size = batch_size
        self.window_size = window_size
        self.epochs = epochs
        self.weight_file = weight_file
        self.model = self.init_network((window_size, window_size, self.bands))
        self.model.compile(loss=dice_coef_loss, optimizer=Adam(learning_rate=0.001),  # Adam(learning_rate=0.001),
                           metrics=['accuracy'])
        self.model = load_model(self.weight_file, custom_objects={'dice_coef_loss': dice_coef_loss})
        self.model.load_weights(self.weight_file)

    def init_network(self, input_size):
        """
        This method initiates the u-network which takes the input_size as initial size of the Input layer
        """
        inputs = Input(input_size)
        conv_1 = Conv2D(16, (3, 3), padding="same", strides=1, activation="relu")(inputs)
        conv_1 = Conv2D(16, (3, 3), padding="same", strides=1, activation="relu")(conv_1)
        pool = MaxPooling2D((2, 2))(conv_1)
        conv_2 = Conv2D(32, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_2 = Conv2D(32, (3, 3), padding="same", strides=1, activation="relu")(conv_2)
        pool = MaxPooling2D((2, 2))(conv_2)
        conv_3 = Conv2D(64, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_3 = Conv2D(64, (3, 3), padding="same", strides=1, activation="relu")(conv_3)
        pool = MaxPooling2D((2, 2))(conv_3)
        conv_4 = Conv2D(128, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_4 = Conv2D(128, (3, 3), padding="same", strides=1, activation="relu")(conv_4)
        pool = MaxPooling2D((2, 2))(conv_4)
        conv_5 = Conv2D(256, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_5 = Conv2D(256, (3, 3), padding="same", strides=1, activation="relu")(conv_5)
        pool = MaxPooling2D((2, 2))(conv_5)
        pool = Dropout(0.5)(pool)

        bn = Conv2D(512, (3, 3), padding="same", strides=1, activation="relu")(pool)
        bn = Conv2D(512, (3, 3), padding="same", strides=1, activation="relu")(bn)

        de_conv_5 = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(bn)
        de_conv_5 = Dropout(0.5)(de_conv_5)
        concat = concatenate([de_conv_5, conv_5])
        de_conv_5 = Conv2D(256, (3, 3), padding="same", activation="relu")(concat)
        de_conv_5 = Conv2D(256, (3, 3), padding="same", activation="relu")(de_conv_5)
        de_conv_4 = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(de_conv_5)
        concat = concatenate([de_conv_4, conv_4])
        de_conv_4 = Conv2D(128, (3, 3), padding="same", activation="relu")(concat)
        de_conv_4 = Conv2D(128, (3, 3), padding="same", activation="relu")(de_conv_4)
        de_conv_3 = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(de_conv_4)
        concat = concatenate([de_conv_3, conv_3])
        de_conv_3 = Conv2D(64, (3, 3), padding="same", activation="relu")(concat)
        de_conv_3 = Conv2D(64, (3, 3), padding="same", activation="relu")(de_conv_3)
        de_conv_2 = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(de_conv_3)
        concat = concatenate([de_conv_2, conv_2])
        de_conv_2 = Conv2D(32, (3, 3), padding="same", activation="relu")(concat)
        de_conv_2 = Conv2D(32, (3, 3), padding="same", activation="relu")(de_conv_2)
        de_conv_1 = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(de_conv_2)
        concat = concatenate([de_conv_1, conv_1])
        de_conv_1 = Conv2D(16, (3, 3), padding="same", activation="relu")(concat)
        de_conv_1 = Conv2D(16, (3, 3), padding="same", activation="relu")(de_conv_1)
        outputs = Conv2D(len(selected_classes), (1, 1), padding="same", activation="softmax")(de_conv_1)
        return Model(inputs=inputs, outputs=[outputs])

    def multi_spectral_image_generator(self, mode='train'):
        """
        This method provides pairs of input RGB images and NIR images and labels as generators
        and can be used for train as well as validation data
        :param mode: str
                can be set for "train" or "validation" data
        """
        # same seed should be in all generators
        SEED = 345
        data_gen_args = dict(rescale=1. / 255)

        X_train_RGB = ImageDataGenerator(**data_gen_args).flow_from_directory(
            str(Path('dataset', mode, 'RGBinputs', 'input').parent), batch_size=self.batch_size, color_mode='rgb',
            seed=SEED)

        X_train_NIR = ImageDataGenerator(**data_gen_args).flow_from_directory(
            str(Path('dataset', mode, 'NIRinputs', 'input').parent), batch_size=self.batch_size, color_mode='rgb',
            seed=SEED)

        # don't rescale masks
        del data_gen_args['rescale']

        y_train = ImageDataGenerator(**data_gen_args).flow_from_directory(
            str(Path('dataset', mode, 'labels', 'label').parent), batch_size=self.batch_size,
            class_mode='input', color_mode='grayscale',
            seed=SEED)

        while True:
            yield np.concatenate((next(X_train_RGB)[0], next(X_train_NIR)[0]), axis=3), np.eye(len(selected_classes))[
                np.squeeze(next(y_train)[0]).astype(int)]

    def train(self):
        checkpoint = ModelCheckpoint(self.weight_file, verbose=1, monitor='val_loss', save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=3,
                                   verbose=0, mode='auto')

        train_gen = self.multi_spectral_image_generator('train')
        val_gen = self.multi_spectral_image_generator('validation')

        _, _, num_of_train = next(os.walk(str(Path('dataset', 'train', 'RGBinputs', 'input'))))
        _, _, num_of_val = next(os.walk(str(Path('dataset', 'validation', 'RGBinputs', 'input'))))

        print('Start training with %d images and %d images for validation' % (len(num_of_train), len(num_of_val)))
        self.history = self.model.fit(train_gen,
                                      steps_per_epoch=len(num_of_train) // self.batch_size,
                                      epochs=self.epochs,
                                      validation_steps=len(num_of_val) // self.batch_size,
                                      validation_data=val_gen,
                                      callbacks=[checkpoint, early_stop])

        with open('history.json', 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

    def test(self):
        """
            Tests the model on the test images in the pre-defined paths in global variables
            then plots a comparison of the prediction and ground truth patches
        """
        x = []
        y = []
        for img_path in list(Path('dataset/test/RGBinputs/input').glob('*.*'))[::5]:
            name = os.path.basename(img_path)
            img_rgb = np.array(Image.open(img_path))
            img_nir = np.array(Image.open(str(img_path).replace('RGBinputs', 'NIRinputs')))
            mask = np.array(
                Image.open(os.path.join(Path('dataset/test/labels/label'), name)))  # read image
            img = np.dstack((img_rgb, img_nir)) * 1.0 / 255
            x.append(img)
            y.append(mask)
        x = np.array(x)
        y = np.array(y)
        self.model.load_weights(self.weight_file)
        output = np.squeeze(self.model.predict(x, verbose=0))
        n_rows = 4
        for i in np.arange(0, output.shape[0], n_rows):
            fig, ax = plt.subplots(n_rows, 3)
            for row in range(n_rows):
                inp = x[i + row, :, :, :3]
                pre = np.argmax(output[i + row, ...], axis=2)
                ori = y[i + row, ...]
                fig.suptitle('Estimation {}'.format(i - row - 1))
                ax[row][0].imshow(inp)
                ax[row][1].imshow(np.array([[colors[int(val)]] for val in pre.reshape(-1)]).reshape(*pre.shape, 3))
                ax[row][2].imshow(np.array([[colors[int(val)]] for val in ori.reshape(-1)]).reshape(*ori.shape, 3))
                if not row:
                    ax[row][0].title.set_text('input')
                    ax[row][1].title.set_text('Estimated')
                    ax[row][2].title.set_text('Ground truth')

                ax[row][0].axis('off')
                ax[row][1].axis('off')
                ax[row][2].axis('off')

            plt.legend(handles=colors_legend, borderaxespad=-15, fontsize='x-small')
            plt.show()

    def estimate_raw_landsat(self, input_landsat_bands_normalized, visual_light_reflectance_mask, metadata, trim=20):
        """
         Estimates the full map image by sliding a window over and
           trimming off sides from each side of 256*256 patch
        :param metadata:
        :param input_landsat_bands_normalized:
        :param visual_light_reflectance_mask:
        :param trim: int
            the number of pixels trimmed of each side of the predicted window
            e.g. 100 -> adds only the middle 56*56 square of the 256*256 patch to the result.
           The trimming is used to avoid creases and artifacts since patch-wise prediction
           has no knowledge of nearby structures from the next patch.
        """

        input_width, input_height, input_band_count = input_landsat_bands_normalized.shape
        in_image = np.reshape(input_landsat_bands_normalized, (1, input_width, input_height, input_band_count))
        result = np.zeros((input_width, input_height))
        for window_data, x, y, x_overflow, y_overflow in get_tiles_generator(input_width, input_height, self.window_size, trim, in_image):
            window = np.zeros((1, self.window_size, self.window_size, self.bands))
            window[:, :window_data.shape[1], :window_data.shape[2], :] = window_data
            output = self.model.predict(window, verbose=0)[:, :window_data.shape[1], :window_data.shape[2], :]
            result[x + trim:x_overflow - trim,
                   y + trim:y_overflow - trim] = np.argmax(output.squeeze()[
                                                    trim:window_data.shape[1] - trim,
                                                    trim:window_data.shape[2] - trim],
                                                    axis=2)
        assert result.shape[0] == input_width and result.shape[1] == input_height
        # ToDo change to log statement
        print(result.shape[0], result.shape[1])
        # skip all pixels without visual light reflectance in landsat scene
        # shift all classes +1 in the result, hence 0:= no data and not no_change
        result += 1
        # ToDo potentially skip this when using numpy masked arrays
        result *= visual_light_reflectance_mask
        temp_result_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif', prefix='landcover_prediction_')
        with rasterio.open(temp_result_file.name, 'w', **metadata) as destination:
            destination.write(result.astype(rasterio.uint8), 1)
        return temp_result_file.name
