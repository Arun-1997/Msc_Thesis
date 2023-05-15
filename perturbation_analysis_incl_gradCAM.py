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
import os

class perturbation_analysis_incl_gradCAM:
    
    def __init__(self):
        self.file_name = "Iowa_2021_july_8448-1792"
        self.img_path = 'Input/sentinel/patches_256/Iowa_July_1_31/test/'+self.file_name+'.tif'
        self.output_root_dir = "Output/perturbation/"
        self.output_path = os.path.join(self.output_root_dir,self.file_name)
        os.makedirs(self.output_path, exist_ok=True)
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
    
    
    def get_saliency_maps(self,method_name,model,input_img,no_channels):
        analyzer = innvestigate.create_analyzer(method_name, model)
        a = analyzer.analyze(input_img)
        a1 = a[0] 
        a1 = a1.sum(axis=np.argmax(np.asarray(a1.shape) == no_channels))
        a1 /= np.max(np.abs(a1))
        return a1
    
    def run_saliency(self):
        lrp_a  = self.get_saliency_maps("lrp.sequential_preset_a_flat", self.mask_cnn_model,self.mask_img_batch,12)
        lrp_a_masked  = self.get_saliency_maps("lrp.sequential_preset_a_flat", self.nomask_cnn_model,self.nomask_img_batch,13)
        lrp_b  = self.get_saliency_maps("lrp.sequential_preset_b_flat", self.mask_cnn_model,self.mask_img_batch,12)
        lrp_b_masked  = self.get_saliency_maps("lrp.sequential_preset_b_flat", self.nomask_cnn_model,self.nomask_img_batch,13)
        gbp  = self.get_saliency_maps("guided_backprop", self.mask_cnn_model,self.mask_img_batch,12)
        gbp_masked  = self.get_saliency_maps("guided_backprop", self.nomask_cnn_model,self.nomask_img_batch,13)
        grad  = self.get_saliency_maps("gradient", self.mask_cnn_model,self.mask_img_batch,12)
        grad_masked  = self.get_saliency_maps("gradient", self.nomask_cnn_model,self.nomask_img_batch,13)
        smoothgrad  = self.get_saliency_maps("smoothgrad", self.mask_cnn_model,self.mask_img_batch,12)
        smoothgrad_masked  = self.get_saliency_maps("smoothgrad", self.nomask_cnn_model,self.nomask_img_batch,13)
        input_t_gradient  = self.get_saliency_maps("input_t_gradient", self.mask_cnn_model,self.mask_img_batch,12)
        input_t_gradient_masked  = self.get_saliency_maps("input_t_gradient", self.nomask_cnn_model,self.nomask_img_batch,13)
        deep_taylor  = self.get_saliency_maps("deep_taylor",self.mask_cnn_model,self.mask_img_batch,12)
        deep_taylor_masked  = self.get_saliency_maps("deep_taylor", self.nomask_cnn_model,self.nomask_img_batch,13)
        integrated_gradients  = self.get_saliency_maps("integrated_gradients", self.mask_cnn_model,self.mask_img_batch,12)
        integrated_gradients_masked  = self.get_saliency_maps("integrated_gradients", self.nomask_cnn_model,self.nomask_img_batch,13)
        self.saliency_dict = {"lrp_a":{"mask":lrp_a_masked,
                         "no_mask":lrp_a},
                "lrp_b":{"mask":lrp_b_masked,
                        "no_mask":lrp_b},
                 "gbp":{"mask":gbp_masked,
                        "no_mask":gbp},
                 "grad":{"mask":grad_masked,
                        "no_mask":grad},
                 "smoothgrad":{"mask":smoothgrad_masked,
                        "no_mask":smoothgrad},
                 "input_t_gradient":{"mask":input_t_gradient_masked,
                        "no_mask":input_t_gradient},
                 "deep_taylor":{"mask":deep_taylor_masked,
                        "no_mask":deep_taylor},
                  "integrated_gradients":{"mask":integrated_gradients_masked,
                        "no_mask":integrated_gradients}
                }
        
        self.set_saliency_plots()
    
    def set_saliency_plots(self):
        grad_cam_mask_file = "Output/saliency_maps/gradCAM_mask_sent/test/"+self.file_name+".tif"
        grad_cam_nomask_file = "Output/saliency_maps/gradCAM_nomask_sent/test/"+self.file_name+".tif"

        grad_cam_mask = reshape_as_image(rasterio.open(grad_cam_mask_file).read()[13:16,:,:])
        grad_cam_nomask = reshape_as_image(rasterio.open(grad_cam_nomask_file).read()[12:15,:,:])
        
        self.saliency_dict["gradCAM"] = {}
        self.saliency_dict["gradCAM"]["mask"] = grad_cam_mask
        self.saliency_dict["gradCAM"]["no_mask"] = grad_cam_nomask
        fig,ax = plt.subplots(9,4,figsize = (10,20))
        ax[0,0].set_title("With Mask Layer")
        ax[0,1].set_title("Without Mask Layer")
        ax[0,2].set_title("Rank (Mask)")
        ax[0,3].set_title("Rank (Without Mask)")

        plot_saliency(ax,0,"lrp_a")
        plot_saliency(ax,1,"lrp_b")
        plot_saliency(ax,2,"gbp")
        plot_saliency(ax,3,"grad")
        plot_saliency(ax,4,"smoothgrad")
        plot_saliency(ax,5,"input_t_gradient")
        plot_saliency(ax,6,"deep_taylor")
        plot_saliency(ax,7,"integrated_gradients")
        plot_saliency(ax,8,"gradCAM",add_dim=True)


        plt.savefig(os.path.join(self.output_path,"Saliency_maps.png"))
        fig.tight_layout()
        plt.close()
    
    def plot_saliency(self,ax,row_no,method_name,add_dim = False):
        ax[row_no,0].set_ylabel(method_name, rotation=90, size='large')
        ax[row_no,0].imshow(self.saliency_dict[method_name]["mask"], cmap="jet")
        # ax[row_no,0].axis("off")
        # ax[row_no,0].set_title("With Mask Layer") 
        ax[row_no,1].imshow(self.saliency_dict[method_name]["no_mask"], cmap="jet")
        # ax[row_no,1].axis("off")
        if add_dim:
            block_size = (64,64,3)
        else:
            block_size = (64,64)

        arr_reduced_mask = block_reduce(self.saliency_dict[method_name]["mask"], block_size=block_size, func=np.mean, cval=np.mean(self.saliency_dict[method_name]["mask"]))
        self.saliency_dict[method_name]["rank_mask"] = arr_reduced_mask

        order_mask = arr_reduced_mask.flatten().argsort()
        ranks_mask = order_mask.argsort()
        self.saliency_dict[method_name]["rank_mask_index"] = ranks_mask
        ax[row_no,2].imshow(arr_reduced_mask, cmap="jet")

        arr_reduced_nomask = block_reduce(self.saliency_dict[method_name]["no_mask"], block_size=block_size, func=np.mean, cval=np.mean(self.saliency_dict[method_name]["no_mask"]))
        self.saliency_dict[method_name]["rank_nomask"] = arr_reduced_nomask

        order_nomask = arr_reduced_nomask.flatten().argsort()
        ranks_nomask = order_nomask.argsort()
        self.saliency_dict[method_name]["rank_nomask_index"] = ranks_nomask
        ax[row_no,3].imshow(arr_reduced_nomask, cmap="jet")
        
        
    def run(self):
        self.run_saliency()

if __name__ == "__main__":
    pp = perturbation_analysis_incl_gradCAM()
    pp.run()