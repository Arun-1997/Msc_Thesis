import os 
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show,reshape_as_raster, reshape_as_image
from itertools import product
import rasterio as rio
from sklearn.preprocessing import StandardScaler
from rasterio import windows,mask
from rasterstats import zonal_stats
import glob,fiona
from rasterio.merge import merge

class DataPreparation:

    def __init__(self):
        
        # To ignore nan value when normalising the values using sklearn#
        np.seterr(divide='ignore', invalid='ignore')
        
        os.chdir("/home/jovyan/MSC_Thesis/MSc_Thesis_2023")
        self.soybean_yield_path = "Input/soybean_yield/soybean_yield_county_level.csv"
        self.county_bdry_path = "Input/county_boundary/county_layer.shp"
        self.outdir = "Output/"
        self.year_list = ['2016','2017','2018','2019','2020','2021']
        # self.year_list = ['2020','2021']
        self.state_name = "Nebraska"
        self.tiles_dir = "Input/sentinel/2021/sent_2021_tiles"
        self.sentinel_image_dir = "Input/sentinel/test_data_from_drive/Msc_Thesis_Data_"+self.state_name+"_60m/"
        self.plot_dir = "Input/plots/"
        # self.sent2_500m_cdl = "Input/sentinel/2021/sent2_2021_500m/sent2_cdl_2021_500m.tif"
        # self.sent2_2021_500m = "Input/sentinel/2021/sent2_2021_500m/MscThesis_sentinel2_2021.tif"
    
        self.target_dir = "Input/Target/"
        # self.CDL_dir = "Input/sentinel/test_data_from_drive/Msc_Thesis_Data_"+self.state_name+"_60m/cdl/"
        self.CDL_dir = "Input/cdl_all_crops/"+self.state_name+"/"

        self.patch_size = 256
        self.scaler = StandardScaler()
        # self.tile_height = 512
        
    def read_input_csv(self):
        # sentinel2 = rasterio.open()
        fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(25,25))
        x_index = 0
        for year in self.year_list:
            soybean_yield_df = gpd.read_file(self.soybean_yield_path)
            crop_yield = soybean_yield_df[soybean_yield_df['Year'] == year]

            county_bdry = gpd.read_file(self.county_bdry_path)
            cnty_renamned = county_bdry.rename(columns={'NAME': 'County'})
            cnty_renamned['County'] = cnty_renamned['County'].str.upper()
            cnty_renamned = cnty_renamned.merge(crop_yield, on='County')
            cnty_renamned = gpd.GeoDataFrame(cnty_renamned, geometry=cnty_renamned['geometry_x'])
            cnty_renamned['Value'] = pd.to_numeric(cnty_renamned['Value'])
            output_path = self.outdir+"/yield_val/yield_"+year+".shp"
            cnty_renamned = cnty_renamned.drop(columns=['geometry_x', 'geometry_y'])
            cnty_renamned.to_file(output_path)
            cnty_renamned.plot(column='Value',legend=True,ax=ax[x_index])
            x_index += 1
        plt.savefig(self.outdir+"yield_plot.png")
        plt.close()
        
        
    def get_ndvi(self,file_path,out_fname):
        
        layer = rio.open(file_path)
        
        ndvi = np.zeros(layer.read(1).shape, dtype=rio.float32)
        bandNIR = layer.read(8)
        bandRed = layer.read(4)
        
        ndvi = (bandNIR.astype(float)-bandRed.astype(float))/(bandNIR.astype(float)+bandRed.astype(float))
        plt.hist(ndvi)
        plt.savefig(self.outdir+out_fname+"hist.png")
        plt.close()
        kwargs = layer.meta
        kwargs.update(
            dtype=rio.float32,
            count=1,
            compress='lzw')

        with rio.open(self.outdir+out_fname+".tif", 'w', **kwargs) as dst:
            dst.write_band(1, ndvi.astype(rio.float32))
        show(ndvi,cmap="Greens")
        plt.savefig(self.outdir+out_fname+".png")
        plt.close()
        
        
    def get_tiles(self, ds, patch_dim):
        
        nols, nrows = ds.meta['width'], ds.meta['height']
        offsets = product(range(0, nols, patch_dim), range(0, nrows, patch_dim))
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off,row_off=row_off,width=patch_dim,height=patch_dim).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform
    
    def get_merged_sentinel(self,year,outpath):
        inp_file_list = glob.glob(outpath+"/*MscThesis_sentinel2*"+str(year)+"*.tif")
        inp_files_layer = [rio.open(i) for i in inp_file_list]
        
        merged_layer = merge(inp_files_layer)
        for i in inp_files_layer:
            i.close()
        
        return merged_layer[0]

    def set_merged_layer(self,year, outpath):
        sent_input = self.get_merged_sentinel(year,outpath)
        # print(sent_input.shape)
        cdl_layer = rio.open(self.CDL_path)
        # Masking of the CDL is done based on the availability of yield value for those counties at that year
        masked_gdf_path = self.outdir+"/yield_val/yield_"+str(year)+".shp"
        with fiona.open(masked_gdf_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        masked_cdl,masked_transform = mask.mask(cdl_layer, shapes, crop=True)
        # print(masked_cdl.shape)
        out_meta = cdl_layer.meta
        out_meta.update({"driver": "GTiff",
                         "height": sent_input.shape[1],
                         "width": sent_input.shape[2],
                         "transform": masked_transform})
        
        masked_cdl_path = self.CDL_dir+"CDL_Soybean_"+self.state_name+"_60m_"+str(year)+"_masked.tif"
        with rio.open(masked_cdl_path, "w", **out_meta) as dest:
            dest.write(masked_cdl)
        cdl_layer.close()        
        cdl_mask_layer = rio.open(masked_cdl_path)
        
        # sent_input_12_bands = sent_input.read()[0:12]
        sent_input_12_bands = sent_input[0:12]

        cdl = cdl_mask_layer.read(1)
        sent_masked = sent_input_12_bands[0]*cdl
        sent_masked = np.nan_to_num(sent_masked,nan=-9999)
        sent_masked[(sent_masked != 0) & (sent_masked!=-9999)] = 1
        sent_masked[sent_masked == -9999] = np.nan
        # sent_masked_norm = self.scaler.fit_transform(sent_masked.reshape(-1, sent_masked.shape[-1])).reshape(sent_masked.shape)
        sent_input_12_bands_img = reshape_as_image(sent_input_12_bands)
        sent_input_13_bands_img = np.dstack((sent_input_12_bands_img,sent_masked))
        sent_input_13_bands = reshape_as_raster(sent_input_13_bands_img)
        
        meta = cdl_mask_layer.meta.copy()
        meta.update(count=13)
        self.merged_out_file = outpath+"/sentinel_merged_"+str(year)+".tif"
        with rio.open(self.merged_out_file, 'w', **meta) as outds:
            outds.write(sent_input_13_bands)
        show(sent_input_13_bands[8],cmap="Greens")
        plt.savefig(os.path.join(self.plot_dir,"sentinel_merged_"+str(year)+".png"))
        plt.close()
        # sent_input.close()
        cdl_mask_layer.close()
        
    def set_masked_layer(self,year, outpath):
        sent_input = self.get_merged_sentinel(year,outpath)
        
        cdl_layer = rio.open(self.CDL_path)
        # Masking of the CDL is done based on the availability of yield value for those counties at that year
        masked_gdf_path = self.outdir+"/yield_val/yield_"+str(year)+".shp"
        with fiona.open(masked_gdf_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        masked_cdl,masked_transform = mask.mask(cdl_layer, shapes, crop=True)
        
        out_meta = cdl_layer.meta
        out_meta.update({"driver": "GTiff",
                         "height": sent_input.shape[1],
                         "width": sent_input.shape[2],
                         "transform": masked_transform})
        
        masked_cdl_path = self.CDL_dir+"CDL_Soybean_"+self.state_name+"_60m_"+str(year)+"_masked.tif"
        with rio.open(masked_cdl_path, "w", **out_meta) as dest:
            dest.write(masked_cdl)
        cdl_layer.close()        
        cdl_mask_layer = rio.open(masked_cdl_path)
        
        # sent_input_12_bands = sent_input.read()[0:12]
        sent_input_12_bands = sent_input[0:12]

        cdl = cdl_mask_layer.read(1)
        sent_masked = sent_input_12_bands*cdl
        sent_masked[sent_masked == 0] = np.nan
        # sent_masked_norm = self.scaler.fit_transform(sent_masked.reshape(-1, sent_masked.shape[-1])).reshape(sent_masked.shape)
        meta = cdl_mask_layer.meta.copy()
        meta.update(count=12)
        self.masked_out_file = outpath+"/sentinel_masked_"+str(year)+".tif"
        with rio.open(self.masked_out_file, 'w', **meta) as outds:
            outds.write(sent_masked)
        show(sent_masked[7],cmap="Greens")
        plt.savefig(os.path.join(self.plot_dir,"sentinel_masked_"+str(year)+".png"))
        plt.close()
        # sent_input.close()
        cdl_mask_layer.close()
        
    def get_tile_patches(self, in_path,out_path ,fname_prefix, patch_dim):

        # in_path = os.path.join(self.sentinel_image_dir,"sent2_2021_Iowa_60m")
        # in_path = glob.glob(in_path+"/*.tif")
        
        # in_path = [self.inp_raster_path,self.CDL_pathCDL]
        # for image in in_path:
        input_filename = in_path
        # out_path = 'Input/sentinel/2021/sent2_2021_Iowa_60m/Iowa_masked_patches/'
        out_path = out_path
        output_filename = fname_prefix+'_{}-{}.tif'
        print(input_filename)
        with rio.open(input_filename) as inds:
            meta = inds.meta.copy()
            for window, transform in self.get_tiles(inds,patch_dim):
                
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                # if fname_prefix== "cdl":
                #     outpath = os.path.join(out_path,output_filename.format(int(window.col_off/2), int(window.row_off/2)))
                # else:
                if np.isnan(inds.read(window=window)).all():
                    continue
                print(window)
                outpath = os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off)))
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))
              
    def get_ndvi_patch(self,patch_src):
        smin=0 
        smax=255

        x = patch_src.read(8) #NIR Band
        # bandNIR = ( x - np.nanmin(x) ) * (smax - smin) / ( np.nanmax(x) - np.nanmin(x) ) + smin
        bandNIR = x
        y = patch_src.read(4) #Red Band
        # bandRed = ( y - np.nanmin(y) ) * (smax - smin) / ( np.nanmax(y) - np.nanmin(y) ) + smin
        bandRed = y
        ndvi = np.zeros(patch_src.read(1).shape, dtype=rio.float32)
        ndvi = ((bandNIR.astype(float)-bandRed.astype(float))/(bandNIR.astype(float)+bandRed.astype(float)))
        avg_ndvi = np.nanmean(ndvi)
        min_ndvi = np.nanmin(ndvi)
        max_ndvi = np.nanmax(ndvi)
        return avg_ndvi,min_ndvi,max_ndvi

    
    def set_target_for_patches(self,state_name,year,output_name):
        yield_inp_gdf = gpd.read_file(self.outdir+"/yield_val/yield_"+str(year)+".shp")
        yield_inp = yield_inp_gdf[yield_inp_gdf['STATE_NAME'] == state_name]
        yield_inp = yield_inp.drop_duplicates(subset='County', keep="first") #Removing duplicates complicates since there are counties with same name in different states. Need to find a better solution to remove duplicates instead of using County column as subset
        #1 Bushel =  27.2 Kg
        # 1 acre is 4046.86 sq. m
        yield_kg_per_sq_m = 27.2/4046.86
        yield_inp["yield_in_kg_per_sqm"] = yield_inp["Value"]*yield_kg_per_sq_m
        
        patch_files_path = self.patchs_masked_output_dir
        patch_files = glob.glob(patch_files_path+"*"+str(year)+"*.tif")
        target_dict = dict()
        patch_name_list = []
        target_yield_list = []
        patch_geom = []
        ndvi_avg = []
        ndvi_min = []
        ndvi_max = []
        # gg = gpd.Geo
        for i_patch in patch_files:
            print(i_patch)
            patch_src = rio.open(i_patch)
            avg_ndvi,min_ndvi,max_ndvi = self.get_ndvi_patch(patch_src)
            
            patch_bounds = list(patch_src.bounds)
            yield_inp_clip = gpd.clip(yield_inp,patch_bounds)
            if len(yield_inp_clip['geometry']) == 0:
                continue
            yield_inp_clip["pixel_count"] = [i["count"] for i in zonal_stats(vectors=yield_inp_clip['geometry'], raster=i_patch, 
                                                                             categorical=False,stats='count')]
            yield_inp_clip["yield_per_county_in_KG"] = yield_inp_clip["yield_in_kg_per_sqm"]*yield_inp_clip["pixel_count"]*3600
            target_yield = yield_inp_clip["yield_per_county_in_KG"].sum()
            patch_name = i_patch.split("/")[-1].split(".")[0]
            patch_name_list.append(patch_name)
            target_yield_list.append(target_yield)
            ndvi_avg.append(avg_ndvi)
            ndvi_min.append(min_ndvi)
            ndvi_max.append(max_ndvi)
            patch_geom.append(box(patch_bounds[0],patch_bounds[1],patch_bounds[2],patch_bounds[3]))
            patch_src.close()

        target_dict["patch_name"] = patch_name_list
        target_dict["year"] = year
        target_dict["yld_kg_sqm"] = target_yield_list
        target_dict["ndvi_avg"] = ndvi_avg
        target_dict["ndvi_max"] = ndvi_max
        target_dict["ndvi_min"] = ndvi_min
        target_dict["geometry"] = patch_geom
        target_gdf = gpd.GeoDataFrame(target_dict,crs="EPSG:4269")
        out_path = os.path.join(self.target_dir,output_name+str(year)+".shp")
        target_gdf.plot(column="yld_kg_sqm",legend=True,cmap="Greens")
        plt.title("Target Yield for patches")
        plt.savefig(os.path.join(self.plot_dir,"target_yield_patches.png"))
        target_gdf.to_file(out_path)
        plt.close()
        
    def run(self):
        # self.read_input_csv()
        # self.read_input_raster()
        # self.get_ndvi(self.sent2_500m_cdl,"NDVI_CDL_2021_500m")
        # self.get_ndvi(self.sent2_2021_500m,"NDVI_Sent2_2021_500m")
        
        # self.patchs_output_dir = "Input/sentinel/patches/Iowa_July_1_31/"
        # self.set_target_for_patches("Iowa",2019,"Iowa")
        
        
        for year in self.year_list:
            cdl_file = self.CDL_dir + "CDL_allCrops_"+self.state_name+"_60m_" + str(year)+".tif"
            out_path = self.CDL_dir + "/patches/"
            self.get_tile_patches(cdl_file,out_path,self.state_name+"_"+year+"_july",self.patch_size)
        
#         for year in self.year_list:
#             self.CDL_path = self.CDL_dir+ "CDL_soybean_"+self.state_name+"_60m_"+year+".tif"
#             self.set_merged_layer(int(year),os.path.join(self.sentinel_image_dir,self.state_name+"_"+year+"/"))
            
#             self.set_masked_layer(int(year),os.path.join(self.sentinel_image_dir,self.state_name+"_"+year+"/"))
#             # self.masked_out_file = os.path.join(self.sentinel_image_dir,"2016_Iowa_july/sentinel_masked_2016.tif")
#             self.patchs_output_dir = self.sentinel_image_dir + "patches/"
#             self.patchs_masked_output_dir = self.sentinel_image_dir + "patches_masked/"
#             self.get_tile_patches(self.merged_out_file,self.patchs_output_dir,self.state_name+"_"+year+"_july",self.patch_size)
#             self.get_tile_patches(self.masked_out_file,self.patchs_masked_output_dir,self.state_name+"_"+year+"_july",self.patch_size)
#             self.set_target_for_patches(self.state_name,int(year),self.state_name)
        
if __name__ == "__main__":
    
    dprep = DataPreparation()
    dprep.run()