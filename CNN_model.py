import os,glob
import numpy as np
import cv2
from numpy import array
import rasterio as rio
from rasterio.plot import reshape_as_image
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPool2D
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Dense
import wandb
from wandb.keras import WandbCallback

from datetime import datetime

class CNN_model:
    
    def __init__(self):
        
        os.chdir("/home/jovyan/MSC_Thesis/MSc_Thesis_2023")
        self.training_path = "Input/sentinel/2021/sent2_2021_Iowa_60m/Iowa_masked_patches/"
        self.target_file_path = "Input/sentinel/2021/Target/Iowa_2021.shp" 
        self.patch_dim = (256, 256, 12)
        self.ignore_patch_list = list()
        self.x = list()
        self.y = list()
        self.set_config()
        self.scaler = StandardScaler()
    
    def set_config(self):
        self.config = {
        "epochs":20,
        "batch_size":64,
        "loss_function":'mse',
        "metrics":['mae'],
        "learning_rate":0.0001
        # "optimizer":'adam'
        }
        wandb.init(project="test-project", entity="msc-thesis",config=self.config)
        now = datetime.now()
        date_time = now.strftime("%d_%m_%Y_%H_%M_%S")

        wandb.run.name = wandb.run.id+"_"+date_time
        
    
    def read_training(self):
        training_file_list = glob.glob(os.path.join(self.training_path,"*.tif"))
        target_gdf = gpd.read_file(self.target_file_path)
        
        for file in training_file_list:
            patch_src = rio.open(file)
            f_name = file.split("/")[-1].split(".")[0]
            patch_src_read = reshape_as_image(patch_src.read())
            if patch_src_read.shape != self.patch_dim:
                self.ignore_patch_list.append(f_name)
                continue
            self.x.append(patch_src_read)
            self.y.append(float(target_gdf.query(f"patch_name == '{f_name}'")["yld_kg_sqm"]))
            patch_src.close()
        self.y = self.scaler.fit_transform(np.array(self.y).reshape(-1, 1))
        self.x = np.array(self.x)
        # print(self.y)
        self.x = np.nan_to_num(self.x, nan=0)# Check for different value for no data
        # print(f"x shape :{self.x.shape}, y shape: {self.y.shape}")
        # print(np.nanmin(self.x),np.nanmax(self.x))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25)
        
    def build_CNN(self):
        ## Set the model architecture here
        
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.patch_dim))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu')) # Add another dense layer
        model.add(layers.Dense(32, activation='relu'))
        
        model.add(layers.Dense(1,activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
                      loss=self.config["loss_function"], metrics=self.config["metrics"])
        
        # early stopping callback
        # es = EarlyStopping(monitor='val_loss',
        #            mode='min',
        #            patience=50,
        #            restore_best_weights = True)
        
        history = model.fit(self.X_train, self.y_train,
                    validation_data = (self.X_test, self.y_test),
                    callbacks=[WandbCallback()],
                    # callbacks=[es],
                    epochs=self.config["epochs"],
                    batch_size=self.config["batch_size"],
                    verbose=1)
#         model.save('Output/models/CNN/')
#         plt.plot(history.history['loss'], label='loss')
#         plt.plot(history.history['val_loss'], label = 'val_loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.ylim([0.5, 1])
#         plt.legend(loc='lower right')

#         # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#         plt.savefig("Output/models/CNN/CNN_loss_acc.png")
#         plt.close()
    
    def run(self):
        self.read_training()
        self.build_CNN()
        # self.prepare
        # pass
        # Plot training and val loss
        # Plot training acc and val acc
        # Correlation map between avergae(min and max) ndvi,evi and bands for patch and the target

        
if __name__ == "__main__":
    cnn = CNN_model()
    cnn.run()