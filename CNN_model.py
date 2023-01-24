import os,glob
import numpy as np
import cv2
from numpy import array
import rasterio as rio
from rasterio.plot import reshape_as_image
import geopandas as gpd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

class CNN_model:
    
    def __init__(self):
        
        os.chdir("/home/jovyan/MSC_Thesis/MSc_Thesis_2023")
        self.training_path = "Input/sentinel/2021/sent2_2021_Iowa_60m/Iowa_masked_patches/"
        self.target_file_path = "Input/sentinel/2021/Target/Iowa_2021.shp" 
        self.patch_dim = (256, 256, 12)
        self.ignore_patch_list = list()
        self.x = list()
        self.y = list()
        
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
        self.x, self.y = np.array(self.x), np.array(self.y)
        print(f"x shape :{self.x.shape}, y shape: {self.y.shape}")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25)
        
    def build_CNN(self):
        
        
    def run(self):
        self.read_training()
        self.build_CNN()
        # self.prepare
        # pass
    

if __name__ == "__main__":
    cnn = CNN_model()
    cnn.run()