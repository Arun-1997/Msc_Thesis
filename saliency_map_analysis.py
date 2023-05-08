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
import xarray as xr
# from tensorflow import keras
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable

class saliency_map_analysis:
    
    def __init__(self):
        self.input = "Output/saliency_maps/gradCAM_nomask_sent/test/"
        self.target_file_path = "Input/Target_256/concat/Iowa.shp"
        self.patch_dim = (256, 256, 15)
        self.output = "Output/saliency_maps_analysis/nomask"
        self.has_mask = False # SET TO False IF MASK LAYER IS NOT IN THE INPUT
    
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
            
            self.saliency = self.get_saliency_band(patch_src_read,file,output_path)
            self.ndvi = self.get_ndvi(patch_src_read,output_path,patch_src.meta)
            self.wdrvi = self.get_wdrvi(patch_src_read,output_path,patch_src.meta)
            self.savi = self.get_savi(patch_src_read,output_path,patch_src.meta)
            self.ndmi = self.get_ndmi(patch_src_read,output_path,patch_src.meta)
            self.evi = self.get_evi(patch_src_read,output_path,patch_src.meta)
            # ccci = self.get_ccci(patch_src_read,output_path,patch_src.meta)
            # ndvi_rgb = self.grayscale_to_rgb(ndvi)
            
            if self.has_mask:
                mask = self.plot_mask(patch_src_read,output_path)                
                self.clip_to_mask(mask,ndvi,evi,output_path)
            
            
            # saliency_array = saliency.flatten()
            # ndvi_array = ndvi.flatten()
            # evi_array = evi.flatten()
            # ndmi_array = ndmi.flatten()
            
            # plt.scatter(evi_array,ndmi_array)
            # plt.title("EVI - NDMI scatter plot")
            # plt.savefig(os.path.join(output_path,"evi_ndmi_scatter.png"))
            # plt.close()
            
            self.get_correlation_plot(self.evi,self.ndmi,output_path,x_label="evi",y_label="ndmi")
            self.get_correlation_plot(self.ndmi,self.wdrvi,output_path,x_label="ndmi",y_label="wdrvi")

            ndvi_diff = np.absolute(self.ndvi - self.saliency)
            
            # square = np.square(ndvi - saliency)
            rmse = np.sqrt(np.average(np.square(self.ndvi-self.saliency)))
            # print("RMSE : ",rmse)
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

    def get_correlation_plot(self, x, y, output_path, x_label="x_label",y_label="y_label"):
                
        x = x.flatten()
        y = y.flatten()
        plt.scatter(x, y, c='crimson',s=2)
        plt.yscale('log')
        plt.xscale('log')

        p1_0 = max(max(x), max(y))
        p2_0 = min(min(x), min(y))
        plt.plot([p1_0, p2_0], [p1_0, p2_0], 'b-')
        plt.xlabel(x_label, fontsize=8)
        plt.ylabel(y_label, fontsize=8)
        plt.axis('equal')
        plt.savefig(os.path.join(output_path,x_label+"_"+y_label+"_corr.png"))
        plt.close()
    
    def clip_to_mask(self,mask,ndvi,evi,output_path):
        # s2_fname = "..."
        # lc_fname = "..."
        # ignore_classes = [4, 5] # Say
        mask_xr = xr.open_rasterio(mask)
        evi_xr = xr.open_rasterio(evi)
        # for my_class in ignore_classes:
        evi_xr_clip = evi_xr.where(mask_xr != 0, other=np.nan)
        plt.imshow(evi_xr_clip)
        plt.savefig(os.path.join(output_path,"evi_clipped.png"))
        plt.close()
    
    def plot_mask(self,patch_src_read,output_path):
        mask_layer = patch_src_read[:,:,12]
        plt.imshow(mask_layer)
        plt.savefig(os.path.join(output_path,"mask_layer.png"))
        plt.close()
        return mask_layer
    
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
    def get_saliency_band(self, patch_src_read,file,output_path):
        
        saliency_bands = patch_src_read[:,:,12:15]
        
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
        
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.title("Saliency Map")

        plt.colorbar(im, cax=cax)
        plt.savefig(output_file_gray)
        plt.close()
        return gray
    
    
    def get_ndmi(self,patch_src_read,output_path,patch_meta):
        # NDMI = (B08 - B11) / (B08 + B11)
        ndmi = np.zeros(patch_src_read[:,:,0].shape, dtype=rio.float32)
        bandNIR = patch_src_read[:,:,7]
        bandSWIR = patch_src_read[:,:,10]
        
        ndmi = (bandNIR.astype(float)-bandSWIR.astype(float))/(bandNIR.astype(float)+bandSWIR.astype(float))
        # print("NDMI min :",ndmi.min())
        # print("NDMI max :",ndmi.max())
        # scaler = MinMaxScaler()
        # scaler.fit(ndvi_original)
        # ndvi = scaler.transform(ndvi_original)
        plt.hist(ndmi)
        plt.savefig(os.path.join(output_path,"ndmi_hist.png"))
        plt.close()
        kwargs = patch_meta
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')
        output_path_ndmi = os.path.join(output_path,"ndmi.tif")
        with rio.open(output_path_ndmi, 'w', **kwargs) as dst:
            dst.write_band(1, ndmi.astype(rio.float32))
        # show(ndvi,cmap="jet")
      
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        im = ax.imshow(ndmi,cmap="jet")
        plt.title("NDMI")
        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(output_path,"ndmi_plot.png"))
        plt.close()
        return ndmi 

    def grayscale_to_rgb(self,gray):
        # save_and_display_gradcam(img_path, heatmap)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * gray)
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((256,256))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        return jet_heatmap
        
    def get_ndvi(self,patch_src_read,output_path,patch_meta):
        
        ndvi = np.zeros(patch_src_read[:,:,0].shape, dtype=rio.float32)
        bandNIR = patch_src_read[:,:,7]
        bandRed = patch_src_read[:,:,3]
        
        ndvi = (bandNIR.astype(float)-bandRed.astype(float))/(bandNIR.astype(float)+bandRed.astype(float))
        # wdrvi = (0.1 * B08 - B04) / (0.1 * B08 + B04);
        # scaler = MinMaxScaler()
        # scaler.fit(ndvi_original)
        # ndvi = scaler.transform(ndvi_original)
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
        plt.title("NDVI")

        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(output_path,"ndvi_plot.png"))
        plt.close()
        return ndvi 

    def get_wdrvi(self,patch_src_read,output_path,patch_meta):
        
        wdrvi = np.zeros(patch_src_read[:,:,0].shape, dtype=rio.float32)
        bandNIR = patch_src_read[:,:,7]
        bandRed = patch_src_read[:,:,3]
        
        wdrvi = (0.1*bandNIR.astype(float)-bandRed.astype(float))/(0.1*bandNIR.astype(float)+bandRed.astype(float))
        # wdrvi = (0.1 * B08 - B04) / (0.1 * B08 + B04);
        plt.hist(wdrvi)
        plt.savefig(os.path.join(output_path,"wdrvi_hist.png"))
        plt.close()
        kwargs = patch_meta
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')
        output_path_wdrvi = os.path.join(output_path,"wdrvi.tif")
        with rio.open(output_path_wdrvi, 'w', **kwargs) as dst:
            dst.write_band(1, wdrvi.astype(rio.float32))
        # show(ndvi,cmap="jet")
      
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        im = ax.imshow(wdrvi,cmap="jet")
        plt.title("WDRVI")

        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(output_path,"wdrvi_plot.png"))
        plt.close()
        return wdrvi
    
    def get_savi(self,patch_src_read,output_path,patch_meta):
        
        savi = np.zeros(patch_src_read[:,:,0].shape, dtype=rio.float32)
        bandNIR = patch_src_read[:,:,7]
        bandRed = patch_src_read[:,:,3]
        L = 0.5
        savi = ((bandNIR.astype(float)-bandRed.astype(float))/(bandNIR.astype(float)+bandRed.astype(float)+L))*(1.0+L)
        # L = soil brightness correction factor could range from (0 -1)
        # index = (B08 - B04) / (B08 + B04 + L) * (1.0 + L); // calculate savi index
        plt.hist(savi)
        plt.savefig(os.path.join(output_path,"savi_hist.png"))
        plt.close()
        kwargs = patch_meta
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')
        output_path_savi = os.path.join(output_path,"savi.tif")
        with rio.open(output_path_savi, 'w', **kwargs) as dst:
            dst.write_band(1, savi.astype(rio.float32))
        # show(ndvi,cmap="jet")
      
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        im = ax.imshow(savi,cmap="jet")
        plt.title("SAVI")

        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(output_path,"savi_plot.png"))
        plt.close()
        return savi
    
    
    def get_ccci(self,patch_src_read,output_path,patch_meta):
        
        # CCCI - Canopy Chlorophyll Content Index
        ccci = np.zeros(patch_src_read[:,:,0].shape, dtype=rio.float32)
        bandNIR = patch_src_read[:,:,7]
        bandRedEdge = patch_src_read[:,:,4]
        bandRed = patch_src_read[:,:,3]
        
        ccci = ((bandNIR.astype(float)-bandRedEdge.astype(float))/(bandNIR.astype(float)+bandRedEdge.astype(float)))/((bandNIR.astype(float)-bandRed.astype(float))/(bandNIR.astype(float)+bandRed.astype(float)))
        plt.hist(ccci)
        plt.savefig(os.path.join(output_path,"ccci_hist.png"))
        plt.close()
        kwargs = patch_meta
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')
        output_path_ccci = os.path.join(output_path,"ccci.tif")
        with rio.open(output_path_ccci, 'w', **kwargs) as dst:
            dst.write_band(1, ccci.astype(rio.float32))
        # show(ndvi,cmap="jet")
      
        ax = plt.subplot()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        im = ax.imshow(ccci,cmap="jet")
        plt.title("Chloro Content Canopy Index")

        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(output_path,"ccci_plot.png"))
        plt.close()
        return ccci
    
  
    def get_evi(self,patch_src_read,output_path,patch_meta):
        
        evi = np.zeros(patch_src_read[:,:,0].shape, dtype=rio.float32)
        bandNIR = patch_src_read[:,:,7]
        bandRed = patch_src_read[:,:,3]
        bandBlue = patch_src_read[:,:,1]
        
        evi = 2.5 * (bandNIR.astype(float)-bandRed.astype(float))/((bandNIR.astype(float)+6.0 * bandRed.astype(float) - 7.5*bandBlue.astype(float)) + 1.0)
        
        # EVI = 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0)
        # scaler1 = MinMaxScaler()
        # scaler1.fit(evi_original)
        # evi = scaler1.transform(evi_original)
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
        plt.title("EVI")

        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(output_path,"evi_plot.png"))
        plt.close()
        return evi 

    
    def upscale_layer(self,layer,scale_no=16):
        pass
    
    def run(self):
        self.read_input()
        
if __name__ == "__main__":
    sal = saliency_map_analysis()
    sal.run()