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
import os,cv2

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
        
        file = rasterio.open(self.img_path)
        self.img_as_raster = file.read()
        img = reshape_as_image(self.img_as_raster)
        self.mask_img_batch = np.expand_dims(img, axis=0)

        self.nomask_img_batch = self.mask_img_batch[:,:,:,0:12]
    
    
    def get_saliency_maps(self,method_name,model,input_img,no_channels):
        analyzer = innvestigate.create_analyzer(method_name, model)
        a = analyzer.analyze(input_img)
        a1 = a[0] 
        a1 = a1.sum(axis=np.argmax(np.asarray(a1.shape) == no_channels))
        a1 /= np.max(np.abs(a1))
        return a1
    
    def run_saliency(self):
        
        lrp_a  = self.get_saliency_maps("lrp.sequential_preset_a_flat", self.nomask_cnn_model,self.nomask_img_batch,12)
        lrp_a_masked  = self.get_saliency_maps("lrp.sequential_preset_a_flat", self.mask_cnn_model,self.mask_img_batch,13)
        lrp_b  = self.get_saliency_maps("lrp.sequential_preset_b_flat", self.nomask_cnn_model,self.nomask_img_batch,12)
        lrp_b_masked  = self.get_saliency_maps("lrp.sequential_preset_b_flat", self.mask_cnn_model,self.mask_img_batch,13)
        gbp  = self.get_saliency_maps("guided_backprop", self.nomask_cnn_model,self.nomask_img_batch,12)
        gbp_masked  = self.get_saliency_maps("guided_backprop", self.mask_cnn_model,self.mask_img_batch,13)
        grad  = self.get_saliency_maps("gradient", self.nomask_cnn_model,self.nomask_img_batch,12)
        grad_masked  = self.get_saliency_maps("gradient", self.mask_cnn_model,self.mask_img_batch,13)
        smoothgrad  = self.get_saliency_maps("smoothgrad", self.nomask_cnn_model,self.nomask_img_batch,12)
        smoothgrad_masked  = self.get_saliency_maps("smoothgrad", self.mask_cnn_model,self.mask_img_batch,13)
        input_t_gradient  = self.get_saliency_maps("input_t_gradient", self.nomask_cnn_model,self.nomask_img_batch,12)
        input_t_gradient_masked  = self.get_saliency_maps("input_t_gradient", self.mask_cnn_model,self.mask_img_batch,13)
        deep_taylor  = self.get_saliency_maps("deep_taylor",self.nomask_cnn_model,self.nomask_img_batch,12)
        deep_taylor_masked  = self.get_saliency_maps("deep_taylor", self.mask_cnn_model,self.mask_img_batch,13)
        integrated_gradients  = self.get_saliency_maps("integrated_gradients", self.nomask_cnn_model,self.nomask_img_batch,12)
        integrated_gradients_masked  = self.get_saliency_maps("integrated_gradients", self.mask_cnn_model,self.mask_img_batch,13)
        
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

        self.plot_saliency(ax,0,"lrp_a")
        self.plot_saliency(ax,1,"lrp_b")
        self.plot_saliency(ax,2,"gbp")
        self.plot_saliency(ax,3,"grad")
        self.plot_saliency(ax,4,"smoothgrad")
        self.plot_saliency(ax,5,"input_t_gradient")
        self.plot_saliency(ax,6,"deep_taylor")
        self.plot_saliency(ax,7,"integrated_gradients")
        self.plot_saliency(ax,8,"gradCAM",add_dim=True)


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
    
    
    def noisy(self,noise_typ,image):
        
        # Parameters
        # ----------
        # image : ndarray
        #     Input image data. Will be converted to float.
        # mode : str
        #     One of the following strings, selecting the type of noise to add:

        #     'gauss'     Gaussian-distributed additive noise.
        #     'poisson'   Poisson-distributed noise generated from the data.
        #     's&p'       Replaces random pixels with 0 or 1.
        #     'speckle'   Multiplicative noise using out = image + n*image,where
        #                 n is uniform noise with specified mean & variance.

        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            # noisy = gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy
        
    def blockshaped(self,arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))

    def unblockshaped(self,arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                   .swapaxes(1,2)
                   .reshape(h, w))
    
    
    def perturbate_based_on_rank(self):
        img_split = []
        img_raster = self.img_as_raster
        for i in img_raster:
            split = self.blockshaped(i,64,64)
            img_split.append(split)
        img_split = np.array(img_split)
        img_split_swap = img_split.swapaxes(0,1)
        gradCAM_rank = self.saliency_dict["gradCAM"]["rank_mask_index"]
        for i in gradCAM_rank:
            # print(img_split_swap[i].shape)
            img_split_swap_image = reshape_as_image(img_split_swap[i])
            img_split_swap_image = self.noisy("gauss",img_split_swap_image)
            img_split_swap[i] = reshape_as_raster(img_split_swap_image)
            # plt.imshow(img_split_swap_image[:,:,8])
            break
        img_split = img_split_swap.swapaxes(0,1)
        img_array = []
        for i in img_split:
            img_reshaped = self.unblockshaped(i,256,256)
            img_array.append(img_reshaped)
        img_array = np.array(img_array)
        # print(img_array.shape)
        plt.imshow(img_array[7,:,:],cmap="jet",vmin=img_raster.min(),vmax=img_raster.max())
        plt.savefig(os.path.join(self.output_path,"perturbation_plot.png"))
        plt.close()
        
    
    def run(self):
        self.run_saliency()
        self.perturbate_based_on_rank()

if __name__ == "__main__":
    pp = perturbation_analysis_incl_gradCAM()
    pp.run()