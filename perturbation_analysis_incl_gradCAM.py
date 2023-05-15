import tensorflow as tf
import innvestigate
import rasterio
import numpy as np
from rasterio.plot import reshape_as_image,reshape_as_raster
import glob
import geopandas as gpd
from tensorflow.keras import models
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
tf.compat.v1.disable_eager_execution()

class perturbation_analysis_incl_gradCAM:
    
    def __init__(self):
        self.file_name = "Iowa_2021_july_8448-1792"
        self.img_path = 'Input/sentinel/patches_256/Iowa_July_1_31/test/'+self.file_name+'.tif'
        self.mask_model_id = "aanaxs4g" # With mask
        self.nomask_model_id = "ezb3xkqf" # No Mask
        self.mask_model_path = glob.glob("wandb/"+ "*"+self.mask_model_id+"*" + "/files/model-best.h5")[0]
        self.nomask_model_path = glob.glob("wandb/"+ "*"+self.nomask_model_id+"*" + "/files/model-best.h5")[0]
        self.mask_ev_gdf = gpd.read_file("Output/Evaluation/"+self.mask_model_id+".shp")
        self.nomask_ev_gdf = gpd.read_file("Output/Evaluation/"+self.nomask_model_id+".shp")

        # print(model_path)
        self.mask_cnn_model = models.load_model(self.mask_model_path)
        self.nomask_cnn_model = models.load_model(self.nomask_model_path)
        self.mask_img_batch = np.expand_dims(img, axis=0)

        self.nomask_img_batch = self.mask_img_batch1[:,:,:,0:12]
    
    
    def run(self):
        pass
    

if __name__ == "__main__":
    pp = perturbation_analysis_incl_gradCAM()
    pp.run()