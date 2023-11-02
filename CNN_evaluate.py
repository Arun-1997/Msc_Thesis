import os,sys,glob
import rasterio as rio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.plot import reshape_as_image
from datetime import datetime
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import wandb
from shapely import wkt


class CNN_evaluate:
    
    def __init__(self):
        os.chdir("/home/jovyan/MSC_Thesis/MSc_Thesis_2023")
        self.test_path = "Input/sentinel/test_data_from_drive/patches_all/normalised_test/"
        self.target_file_path = "Input/Target/concat/target_yield.shp"
        self.output_eval_dir = "Output/Evaluation/"
        self.pred_val = list()
        self.test_patches = list()
        self.true_val = list()
        self.patch_name_list = list()
        self.patch_geom_list = list()
        self.patch_dim = (256, 256, 12) # Change if you want to include include mask layer (
        # 13 for mask layer
        # 12 for only the bands
        wandb.init(project="CNN_model_evaluate", entity="msc-thesis")
        now = datetime.now()
        date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
        # self.model_id = "leuo8izn" # With mask all states
        # self.model_id = "vosvg9hw" # No mask all states
        # self.model_id = "vdv48shg" # NO mask all states 3 June 2023
        # self.model_id = "ezb3xkqf" # No Mask
        self.model_id = "tpd4yg53" # Inception run 3
        wandb.run.name = self.model_id+"_eval_"+date_time
        
    def read_test(self):
        test_file_list = glob.glob(os.path.join(self.test_path,"*.tif"))
        target_gdf = gpd.read_file(self.target_file_path)
        print("Total Number of Patches:",len(test_file_list))
        pred_df = dict()
        count = 0 
        
        for file in test_file_list:
            
            patch_src = rio.open(file)
            f_name = file.split("/")[-1].split(".")[0]
            patch_src_read = reshape_as_image(patch_src.read()[0:12]) #Change if want to include mask layer
            if patch_src_read.shape != self.patch_dim:
                # self.ignore_patch_list.append(f_name)
                # print("Patch Dimensions Mismatch, skipping patch : {}".format(f_name))
                continue                
            if np.isnan(patch_src_read).any():
                # print("Has Nan values, skipping patch : {}".format(f_name))
                continue
            
            rec= target_gdf.query(f"patch_name == '{f_name}'")
            query = rec["ykg_by_e7"]
            if len(query) != 1:
                # print("patch has no target value, skipping patch : {}".format(f_name))
                continue
            # self.test_patches.append(patch_src_read)  
            
            self.patch_name_list.append(f_name)
            self.patch_geom_list.append(rec["geometry"].iloc[0])
            
            self.run_prediction(patch_src_read)
            self.true_val.append(float(query))
            patch_src.close()
            # count+=1
            # if count == 10:
            #     break
        
        
        self.test_patches = np.array(self.test_patches)        
        self.true_val = np.array(self.true_val)
        pred_df["patch_name"] = self.patch_name_list
        pred_df["true_val"] = self.true_val
        pred_df[self.model_id] = self.pred_val
        wandb.log({"table": pd.DataFrame(pred_df)})

        pred_df["geometry"] = self.patch_geom_list
        # df['Coordinates'] = geopandas.GeoSeries.from_wkt(df['Coordinates'])

        pred_gdf = gpd.GeoDataFrame(pred_df,crs="EPSG:4269")
        pred_gdf.to_file(os.path.join(self.output_eval_dir,self.model_id+".shp"))

    def run_prediction(self,patch_src_read):
        img_array = image.img_to_array(patch_src_read)
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_batch)
        self.pred_val.append(prediction[0][0])
        self.test_patches.append(img_batch)

    
    def run(self):
        
        model_path = glob.glob("wandb/"+ "*"+self.model_id+"*" + "/files/model-best.h5")[0]
        
        # print(model_path)
        self.model = models.load_model(model_path)

        self.read_test()
        

if __name__ == "__main__":
    pred = CNN_evaluate()
    pred.run()