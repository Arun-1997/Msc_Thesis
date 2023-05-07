import os,glob
import numpy as np
import cv2
from numpy import array
import rasterio as rio
from PIL import Image
import matplotlib.image as mpimg
from rasterio.plot import reshape_as_image,reshape_as_raster,show
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cm as cm
# from tensorflow import keras
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable

class saliency_map_analysis:
    
    def __init__(self):
        self.input = "Output/saliency_maps/gradCAM_mask_sent/test/"
        self.target_file_path = "Input/Target_256/concat/Iowa.shp"
        self.patch_dim = (256, 256, 16)
        self.output = "Output/saliency_maps_analysis/mask"
    
    def read_input(self):
        input_file_list = glob.glob(os.path.join(self.input,"*.tif"))
        target_gdf = gpd.read_file(self.target_file_path)
        count = 0
        for file in input_file_list:
            
            patch_src = rio.open(file)
            f_name = file.split("/")[-1].split(".")[0]
            output_path = os.path.join(self.output,f_name)
            os.makedirs(output_path, exist_ok=True)
            patch_src_read = reshape_as_image(patch_src.read())
            if patch_src_read.shape != self.patch_dim:
                continue
                
            if np.isnan(patch_src_read).any():
                continue
            
            query = target_gdf.query(f"patch_name == '{f_name}'")["ykg_by_e7"]
            if len(query) != 1:
                continue
        
            saliency = self.get_saliency_band(patch_src_read,file,output_path)
            ndvi = self.get_ndvi(patch_src_read,output_path,patch_src.meta)
            evi = self.get_evi(patch_src_read,output_path,patch_src.meta)
            saliency_array = saliency.flatten()
            ndvi_array = ndvi.flatten()
 
            plt.scatter(ndvi_array,saliency_array)
            plt.savefig(os.path.join(output_path,"sal_ndvi_scatter.png"))
            plt.close()
            
            
            
            ndvi_diff = np.absolute(ndvi - saliency)
            
            # square = np.square(ndvi - saliency)
            rmse = np.sqrt(np.average(np.square(ndvi-saliency)))
            print("RMSE : ",rmse)
            # plt.imshow(ndvi_diff)
            ax = plt.subplot()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(ndvi_diff,cmap="RdBu")

            plt.colorbar(im, cax=cax)
            plt.savefig(os.path.join(output_path,"sal_ndvi_absdiff.png"))
            plt.close()

            
            count +=1
            
            
            break
    
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
    def get_saliency_band(self, patch_src_read,file,output_path):
        
        saliency_bands = patch_src_read[:,:,13:16]
        
        output_file_rgb = os.path.join(output_path,"saliency_rgb.png")
       
        output_file_gray = os.path.join(output_path,"saliency_gray.png")
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(saliency_bands,cmap="jet",vmin=0,vmax=1)
        
        plt.colorbar(im, cax=cax)
        plt.savefig(output_file_rgb)
        plt.close()
        # img = Image.open().convert('L')
        # img.save(output_file_gray)
        gray = self.rgb2gray(saliency_bands)
        plt.hist(gray)
        plt.savefig(os.path.join(output_path,"saliency_map_hist.png"))
        plt.close()
        im = plt.imshow(gray, cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
        
        plt.colorbar(im, cax=cax)
        plt.savefig(output_file_gray)
        plt.close()
        return gray
    
    def get_ndvi(self,patch_src_read,output_path,patch_meta):
        
        ndvi = np.zeros(patch_src_read[:,:,0].shape, dtype=rio.float32)
        bandNIR = patch_src_read[:,:,7]
        bandRed = patch_src_read[:,:,3]
        
        ndvi_original = (bandNIR.astype(float)-bandRed.astype(float))/(bandNIR.astype(float)+bandRed.astype(float))
        
        # EVI = 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0)
        scaler = MinMaxScaler()
        scaler.fit(ndvi_original)
        ndvi = scaler.transform(ndvi_original)
        plt.hist(ndvi)
        plt.savefig(os.path.join(output_path,"ndvi_hist.png"))
        plt.close()
        kwargs = patch_meta
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')
        output_path_ndvi = os.path.join(output_path,"ndvi.tif")
        with rio.open(output_path_ndvi, 'w', **kwargs) as dst:
            dst.write_band(1, ndvi.astype(rio.float32))
        # show(ndvi,cmap="jet")
      
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        im = ax.imshow(ndvi,cmap="jet",vmin=0,vmax=1)
        
        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(output_path,"ndvi_plot.png"))
        plt.close()
        return ndvi 

    def get_evi(self,patch_src_read,output_path,patch_meta):
        
        evi = np.zeros(patch_src_read[:,:,0].shape, dtype=rio.float32)
        bandNIR = patch_src_read[:,:,7]
        bandRed = patch_src_read[:,:,3]
        bandBlue = patch_src_read[:,:,1]
        
        evi = 2.5 * (bandNIR.astype(float)-bandRed.astype(float))/((bandNIR.astype(float)+6.0 * bandRed.astype(float) - 7.5*bandBlue.astype(float)) + 1.0)
        
        # EVI = 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0)
        # scaler = MinMaxScaler()
        # scaler.fit(ndvi_original)
        # ndvi = scaler.transform(ndvi_original)
        plt.hist(evi)
        plt.savefig(os.path.join(output_path,"evi_hist.png"))
        plt.close()
        kwargs = patch_meta
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')
        output_path_evi = os.path.join(output_path,"evi.tif")
        with rio.open(output_path_evi, 'w', **kwargs) as dst:
            dst.write_band(1, evi.astype(rio.float32))
        # show(ndvi,cmap="jet")
      
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        im = ax.imshow(evi,cmap="jet")
        
        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(output_path,"evi_plot.png"))
        plt.close()
        return evi 

    
    def run(self):
        self.read_input()
        
if __name__ == "__main__":
    sal = saliency_map_analysis()
    sal.run()