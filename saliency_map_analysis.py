import os,glob
import numpy as np
import cv2
from numpy import array
import rasterio as rio
from rasterio.plot import reshape_as_image,reshape_as_raster
import geopandas as gpd
# from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cm as cm
# from tensorflow import keras
from mpl_toolkits.axes_grid1 import make_axes_locatable

class saliency_map_analysis:
    
    def __init__(self):
        self.input = "Output/saliency_maps/gradCAM_sent/test/"
        self.target_file_path = "Input/Target_256/concat/Iowa.shp"
        self.patch_dim = (256, 256, 15)
        self.output = "Output/saliency_maps_analysis/"
    
    def read_input(self):
        input_file_list = glob.glob(os.path.join(self.input,"*.tif"))
        target_gdf = gpd.read_file(self.target_file_path)
        count = 0
        for file in input_file_list:
            patch_src = rio.open(file)
            f_name = file.split("/")[-1].split(".")[0]
            patch_src_read = reshape_as_image(patch_src.read())
            if patch_src_read.shape != self.patch_dim:
                continue
                
            if np.isnan(patch_src_read).any():
                continue
            
            query = target_gdf.query(f"patch_name == '{f_name}'")["ykg_by_e7"]
            if len(query) != 1:
                continue
            
            self.get_saliency_band(patch_src_read,f_name)
            count +=1
            
            
            break

    def get_saliency_band(self, patch_src_read,f_name):
        
        saliency_bands = patch_src_read[:,:,12:15]
        print(saliency_bands.shape)
        output_file = os.path.join(self.output,f_name+".png")
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(saliency_bands,cmap="jet")
        plt.colorbar(im, cax=cax)
        plt.savefig(output_file)
        
    def run(self):
        self.read_input()
        
if __name__ == "__main__":
    sal = saliency_map_analysis()
    sal.run()