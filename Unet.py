# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python
#     language: python
#     name: python3
# ---

# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.plot import reshape_as_image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split
import glob,os,sys,cv2
from datetime import datetime

import wandb
from wandb.keras import WandbCallback

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
training_path = "Input/sentinel/test_data_from_drive/patches_all/normalised_train"
target_file_path = "Input/Target/concat/target_yield.shp"
patch_dim = (256, 256, 13)
model_path = "wandb/run-20231105_115239-3eexbosr/files/model-best.h5"

# +
metrics = [tf.keras.metrics.BinaryIoU(target_class_ids = (0, 1),
                            threshold=0.5,
                            name=None,
                            dtype=None),tf.keras.metrics.BinaryAccuracy(
                            name='binary_accuracy', dtype=None, threshold=0.5
                        )]


config = {
"epochs":30,
"batch_size":400,
"loss_function":'binary_crossentropy',
# "metrics":[mse,rmse,mae,mape,msle,cos_sim,log_cos],
"metrics":metrics,
"learning_rate":1e-4
# "optimizer":'adam'
}
wandb.init(project="unet_segmentation", entity="msc-thesis",config=config)
now = datetime.now()
date_time = now.strftime("%d_%m_%Y_%H_%M_%S")

wandb.run.name = wandb.run.id+"_"+date_time


# +
def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


# +
def read_training():
    training_file_list = glob.glob(os.path.join(training_path,"*.tif"))
    target_gdf = gpd.read_file(target_file_path)
    print("Total Number of Patches:",len(training_file_list))
    ignore_patch_list = list()
    x = list()
    y = list()
    X_train = list()
    X_test = list()
    y_train = list()
    y_test = list()
    count = 0 
    for file in training_file_list:

        patch_src = rio.open(file)
        f_name = file.split("/")[-1].split(".")[0]
        patch_src_read = reshape_as_image(patch_src.read()) ## Change the index here to add or remove the mask layer
        # print(0)
        if patch_src_read.shape != patch_dim:
            ignore_patch_list.append(f_name)
            # print("Patch Dimensions Mismatch, skipping patch : {}".format(f_name))
            continue

        # print(1)
        if np.isnan(patch_src_read).any():
            # print("Has Nan values, skipping patch : {}".format(f_name))
            continue

        # print(2)
        query = target_gdf.query(f"patch_name == '{f_name}'")["ykg_by_e7"]
        if len(query) != 1:
            # print("patch has no target value, skipping patch : {}".format(f_name))
            continue
        # print(patch_src_read[:,:,0:12].shape)
        # print(patch_src_read[:,:,12].shape)

        x.append(patch_src_read[:,:,0:12])
        y.append(patch_src_read[:,:,12])
        # y.append(float(query))

        patch_src.close()
        # print(count)
        count +=1
        # if count > 100:
        #     break

    # self.y = self.scaler.fit_transform(np.array(self.y).reshape(-1, 1))
    y = np.array(y)
    y = np.expand_dims(y,-1)
    x = np.array(x)
    
    # x = (x-np.min(x))/(np.max(x)-np.min(x))
    print("Any Null values? ",np.isnan(x).any())
    # print(self.y)
    # self.x = np.nan_to_num(self.x, nan=0)# Check for different value for no data
    print(f"x shape :{x.shape}, y shape: {y.shape}")
    # print(np.nanmin(self.x),np.nanmax(self.x))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return X_train, X_test, y_train, y_test
    #Also, split the training into train and val
    # For testing, keep a part of the dataset as seperate (final month)


# -



# +
def build_unet_model():
    # inputs
    inputs = layers.Input(shape=(256,256,12))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    
    # flat = layers.Flatten()
    # model.add(layers.Dense(64, activation='relu')) # Add another dense layer
    # model.add(Dropout(0.5))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(16, activation='relu'))
    # model.add(layers.Dense(8, activation='relu'))
    # model.add(layers.Dense(4, activation='relu'))
    # model.add(layers.Dense(1,activation='linear'))
    
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    # outputs = layers.Conv2D(1, 1, padding="same", activation = "softmax")(u9)
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


# unet_model = build_unet_model()
# -




# +
def tf_parse(x,y):
    def f(x, y):
        return x, y
    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([256, 256, 12])
    masks.set_shape([256, 256, 1])
    return images, masks


def tf_dataset(x,y,batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # break
    return dataset


X_train, X_test, y_train, y_test = read_training()
unet_model = models.load_model(model_path, compile=True)

train_dataset = tf_dataset(X_train,y_train,batch=config["batch_size"])
validation_dataset = tf_dataset(X_test,y_test,batch=config["batch_size"])
# -

# +
# unet_model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
#                    loss=config["loss_function"], metrics=config["metrics"])

callbacks = [
    ModelCheckpoint("unet_multi_output/model_BSize_"+str(config["batch_size"])+"_NEpochs_"+str(config["epochs"])+"_"+str(datetime.now())+".h5"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2),
    CSVLogger("unet_multi_output/data_BSize_"+str(config["batch_size"])+"_NEpochs_"+str(config["epochs"])+"_"+str(datetime.now())+".csv")
    # TensorBoard(),
    # EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=False)
]
# -

model_history = unet_model.fit(train_dataset,
                              validation_data = validation_dataset,
                              epochs=config["epochs"],
                              callbacks=[WandbCallback(),callbacks])
                            



# +
