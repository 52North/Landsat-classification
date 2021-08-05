import os
import pickle
from pathlib import Path
import rasterio
from PIL import Image
from matplotlib import pyplot as plt, patches
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Conv2DTranspose, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from preprocessing.image_registration import rotate_datasets

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

colors = [(0, 0, 0)] + list(plt.cm.get_cmap('Paired').colors)
colors_legend = [patches.Patch(color=colors[i], label=classes_names[i]) for i in range(len(colors))]

class UNET:
    def __init__(self, batch_size=64, epochs=30, window_size=256):
        self.bands = 6
        self.batch_size = batch_size
        self.window_size = window_size
        self.epochs = epochs
        self.weight_file = str(Path('best_weight.hdf5'))
        self.model = self.init_network((window_size, window_size, self.bands))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001),
                           metrics=['accuracy'])

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
        pool = Dropout(0.2)(pool)
        conv_3 = Conv2D(64, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_3 = Conv2D(64, (3, 3), padding="same", strides=1, activation="relu")(conv_3)
        pool = MaxPooling2D((2, 2))(conv_3)
        conv_4 = Conv2D(128, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_4 = Conv2D(128, (3, 3), padding="same", strides=1, activation="relu")(conv_4)
        pool = MaxPooling2D((2, 2))(conv_4)
        conv_5 = Conv2D(256, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_5 = Conv2D(256, (3, 3), padding="same", strides=1, activation="relu")(conv_5)
        pool = MaxPooling2D((2, 2))(conv_5)
        bn = Conv2D(512, (3, 3), padding="same", strides=1, activation="relu")(pool)
        bn = Conv2D(512, (3, 3), padding="same", strides=1, activation="relu")(bn)
        de_conv_5 = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(bn)
        concat = concatenate([de_conv_5, conv_5])
        de_conv_5 = Conv2D(256, (3, 3), padding="same", activation="relu")(concat)
        de_conv_5 = Conv2D(256, (3, 3), padding="same", activation="relu")(de_conv_5)
        de_conv_4 = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(de_conv_5)
        concat = concatenate([de_conv_4, conv_4])
        de_conv_4 = Conv2D(128, (3, 3), padding="same", activation="relu")(concat)
        de_conv_4 = Conv2D(128, (3, 3), padding="same", activation="relu")(de_conv_4)
        de_conv_3 = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(de_conv_4)
        concat = concatenate([de_conv_3, conv_3])
        concat = Dropout(0.1)(concat)
        de_conv_3 = Conv2D(64, (3, 3), padding="same", activation="relu")(concat)
        de_conv_3 = Conv2D(64, (3, 3), padding="same", activation="relu")(de_conv_3)
        de_conv_2 = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(de_conv_3)
        concat = concatenate([de_conv_2, conv_2])
        concat = Dropout(0.3)(concat)
        de_conv_2 = Conv2D(32, (3, 3), padding="same", activation="relu")(concat)
        de_conv_2 = Conv2D(32, (3, 3), padding="same", activation="relu")(de_conv_2)
        de_conv_1 = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(de_conv_2)
        concat = concatenate([de_conv_1, conv_1])
        concat = Dropout(0.5)(concat)
        de_conv_1 = Conv2D(16, (3, 3), padding="same", activation="relu")(concat)
        de_conv_1 = Conv2D(16, (3, 3), padding="same", activation="relu")(de_conv_1)
        outputs = Conv2D(len(classes_names), (1, 1), padding="same", activation="softmax")(de_conv_1)
        return Model(inputs=inputs, outputs=[outputs])

    def multi_spectral_image_generator(self, mode='train'):
        """
        This method provides pairs of input RGB images and NIR images and labels as generators
        and can be used for train as well as validation data
        :param mode: str
                can be set for "train" or "validation" data
        """
        # same seed should be in all generators
        SEED = 213
        data_gen_args = dict(rescale=1. / 255)
        # apply augmentation only for train-data
        if mode == 'train':
            data_gen_args = dict(
                rescale=1. / 255,
                horizontal_flip=True,
                vertical_flip=True
            )

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
            yield np.concatenate((next(X_train_RGB)[0], next(X_train_NIR)[0]), axis=3), next(y_train)[0].astype(float)

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
        for img_path in list(Path('dataset/test/RGBinputs/input').glob('*.*'))[::20]:
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

    def estimate_raw_landsat(self, path: str, trim=20):
        """
         Estimates the full map image by sliding a window over and
           trimming off sides from each side of 256*256 patch
        :param path:
             folder path of landsat scene
        :param trim: int
            the number of pixels trimmed of each side of the predicted window
            e.g. 100 -> adds only the middle 56*56 square of the 256*256 patch to the result.
           The trimming is used to avoid creases and artifacts since patch-wise prediction
           has no knowledge of nearby structures from the next patch.
        """
        multi_image = [rasterio.open(band_path) for band_path in list(Path(path).glob('*SR_B[2-7].TIF'))]
        profile = multi_image[0].meta.copy()
        profile.update(count=7)
        with rasterio.open(Path(path, '%s.tif' % 'merged'), 'w',
                           **profile) as dst:
            for i, band in enumerate(multi_image, start=2):
                dst.write(band.read(1), i)
        input_map = rotate_datasets(Path(path, 'merged.tif'), label=False)
        self.model.load_weights(self.weight_file)
        w, h, _ = input_map.shape
        stepSize = self.window_size - trim * 2
        in_image = np.reshape(input_map, (1, input_map.shape[0], input_map.shape[1], input_map.shape[2]))
        res = np.zeros((input_map.shape[0], input_map.shape[1]))
        for y in range(0, input_map.shape[1], stepSize):
            for x in range(0, input_map.shape[0], stepSize):
                x_overflow = x + self.window_size
                y_overflow = y + self.window_size
                if x_overflow > w and y_overflow > h:
                    x_overflow = w-1
                    y_overflow = h-1
                elif x_overflow > w:
                    x_overflow = w-1
                elif y_overflow > h:
                    y_overflow = h-1
                window_data = in_image[:, x:x_overflow, y:y_overflow, :]
                window = np.zeros((1, self.window_size, self.window_size, self.bands))
                window[:, :window_data.shape[1], :window_data.shape[2], :] = window_data
                output_all = self.model.predict(window, verbose=0)
                output = output_all[:,:window_data.shape[1],:window_data.shape[2],:]
                output = (output - np.amin(output)) * 1.0 / (np.amax(output) - np.amin(output))
                res[x + trim:x_overflow - trim,
                y + trim:y_overflow - trim] = np.argmax(output.squeeze()[
                                                                  trim:window_data.shape[1] - trim,
                                                                  trim:window_data.shape[2] - trim],
                                                                  axis=2)
        assert res.shape[0] == w and res.shape[1] == h
        print(res.shape[0], res.shape[1])

        plt.imshow(np.array([[colors[int(val)]] for val in res.reshape(-1)]).reshape((w, h, 3)))
        plt.legend(handles=colors_legend, borderaxespad=-15, fontsize='x-small')
        plt.show()

        out_im = Image.fromarray(res)
        return np.array(out_im)
