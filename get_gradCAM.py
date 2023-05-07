import os,glob
import numpy as np
import cv2
from numpy import array
import rasterio as rio
from rasterio.plot import reshape_as_image,reshape_as_raster
import geopandas as gpd
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cm as cm
from tensorflow import keras
class get_gradCAM:
    
    def __init__(self):
        os.chdir("/home/jovyan/MSC_Thesis/MSc_Thesis_2023")
        self.training_path = "Input/sentinel/patches_256/Iowa_July_1_31/test/"
        self.target_file_path = "Input/Target_256/concat/Iowa.shp"
        self.model_id = "aanaxs4g" # With mask
        # self.model_id = "ezb3xkqf" # No Mask
        self.patch_dim = (256, 256, 13)
        self.output_path = "Output/saliency_maps/gradCAM_mask_sent/test/"
    
    
    def read_training(self):
        training_file_list = glob.glob(os.path.join(self.training_path,"*.tif"))
        target_gdf = gpd.read_file(self.target_file_path)
        print("Total Number of Patches:",len(training_file_list))
        
        count = 0 
        for file in training_file_list:

            patch_src = rio.open(file)
            f_name = file.split("/")[-1].split(".")[0]
            patch_src_read = reshape_as_image(patch_src.read()[0:13]) ## Change the index here to add or remove the mask layer
            if patch_src_read.shape != self.patch_dim:               
                continue
                
            if np.isnan(patch_src_read).any():
                continue
            
            query = target_gdf.query(f"patch_name == '{f_name}'")["ykg_by_e7"]
            if len(query) != 1:
                continue
            
            count +=1
            # if count >= 2:
            #     break
            jet_heatmap = self.run_gradCAM(patch_src_read)
            out_meta = patch_src.meta.copy()
            out_meta.update(
                dtype=rio.float32,
                count=16,
                )
            
            jet_heatmap = jet_heatmap/255
            sent_jet_heatmap = np.dstack((patch_src_read,jet_heatmap))
            jet_heatmap_raster = reshape_as_raster(sent_jet_heatmap)
            # print(jet_heatmap_raster.shape)
            print(f_name)
            out_file = self.output_path+f_name+".tif"
            with rio.open(out_file, 'w', **out_meta) as outds:
                outds.write(jet_heatmap_raster)
            patch_src.close()
        
        print(f"Count : {count}") 


    
    def run_gradCAM(self,patch_src_read):
        img_array = image.img_to_array(patch_src_read)
        img_batch = np.expand_dims(img_array, axis=0)
        # Generate class activation heatmap
        last_conv_layer_name = "conv2d_2"
        heatmap = self.make_gradcam_heatmap(img_batch, self.model, last_conv_layer_name)
        jet_heatmap = self.get_rescaled_heatmap(heatmap)
        return jet_heatmap
    
    
    
    def get_rescaled_heatmap(self,heatmap):
        # save_and_display_gradcam(img_path, heatmap)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((256,256))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        return jet_heatmap
        
    def make_gradcam_heatmap(self,img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
        
                    
    
    def run(self):
        model_path = glob.glob("wandb/"+ "*"+self.model_id+"*" + "/files/model-best.h5")[0]
        # print(model_path)
        self.model = models.load_model(model_path)
        self.read_training()
        
if __name__ == "__main__":
    grad = get_gradCAM()
    grad.run()