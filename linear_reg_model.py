import os,glob
import numpy as np
import cv2
from numpy import array
import rasterio as rio
from rasterio.plot import reshape_as_image
import geopandas as gpd
# from sklearn.preprocessing import StandardScaler
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
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from datetime import datetime

# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
# sess = tf.Session(config=config)


# TRY USING CPU INSTEAD OF GPU?
class linear_reg_model:
    
    def __init__(self):
        
        os.chdir("/home/jovyan/MSC_Thesis/MSc_Thesis_2023")
        self.training_path = "Input/sentinel/test_data_from_drive/patches_all/train/"
        self.target_file_path = "Input/Target/concat/target_yield.shp"
        self.patch_dim = (256, 256, 13)
        self.ignore_patch_list = list()
        self.x = list()
        self.y = list()
        self.set_config()
        # self.scaler = StandardScaler()
    
    def set_config(self):
        self.config = {
        "epochs":100,
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
        print("Total Number of Patches:",len(training_file_list))
        
        count = 0 
        for file in training_file_list:

            patch_src = rio.open(file)
            f_name = file.split("/")[-1].split(".")[0]
            patch_src_read = reshape_as_image(patch_src.read()) ## Change the index here to add or remove the mask layer
            if patch_src_read.shape != self.patch_dim:
                self.ignore_patch_list.append(f_name)
                # print("Patch Dimensions Mismatch, skipping patch : {}".format(f_name))
                continue
                
            if np.isnan(patch_src_read).any():
                # print("Has Nan values, skipping patch : {}".format(f_name))
                continue
            
            query = target_gdf.query(f"patch_name == '{f_name}'")["ykg_by_e7"]
            if len(query) != 1:
                # print("patch has no target value, skipping patch : {}".format(f_name))
                continue
            self.x.append(patch_src_read)
            self.y.append(float(query))
            patch_src.close()
            count +=1
            # if count > 5000:
            #     break

        # self.y = self.scaler.fit_transform(np.array(self.y).reshape(-1, 1))
        self.y = np.array(self.y)
        self.x = np.array(self.x)
        print("Any Null values? ",np.isnan(self.x).any())
        # print(self.y)
        # self.x = np.nan_to_num(self.x, nan=0)# Check for different value for no data
        print(f"x shape :{self.x.shape}, y shape: {self.y.shape}")
        # print(np.nanmin(self.x),np.nanmax(self.x))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25)
        
        #Also, split the training into train and val
        # For testing, keep a part of the dataset as seperate (final month)
    def build_OLS(self):
        ## Set the model architecture here
        
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)

        
    def run(self):
        self.read_training()
        self.build_OLS()
        # self.prepare
        # pass
        # Plot training and val loss
        # Plot training acc and val acc
        # Correlation map between avergae(min and max) ndvi,evi and bands for patch and the target

        
if __name__ == "__main__":
    ols = linear_reg_model()
    ols.run()