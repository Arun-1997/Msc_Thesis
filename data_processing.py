import os 
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
from itertools import product
import rasterio as rio
from sklearn.preprocessing import StandardScaler
from rasterio import windows,mask
from rasterstats import zonal_stats
import glob,fiona


class DataPreparation:

    def __init__(self):
        os.chdir("/home/jovyan/MSC_Thesis/MSc_Thesis_2023")
        self.soybean_yield_path = "Input/soybean_yield/soybean_yield_county_level.csv"
        self.county_bdry_path = "Input/county_boundary/county_layer.shp"
        self.outdir = "Output/"
        self.year_list = ['2015','2016','2017','2018','2019','2020','2021']
        self.tiles_dir = "Input/sentinel/2021/sent_2021_tiles"
        self.sentinel_image_dir = "Input/sentinel/2021"
        self.plot_dir = "Input/plots/"
        # self.sent2_500m_cdl = "Input/sentinel/2021/sent2_2021_500m/sent2_cdl_2021_500m.tif"
        # self.sent2_2021_500m = "Input/sentinel/2021/sent2_2021_500m/MscThesis_sentinel2_2021.tif"
        self.inp_raster_path = os.path.join(self.sentinel_image_dir,"sent2_2021_Iowa_60m/sentinel2_Iowa_60m_clipped.tif")
        
        self.CDL_path = "Input/cdl/Iowa_60m/CDL_Soybean_Iowa_60m_2021_clipped.tif"
        self.CDL_dir = "Input/cdl/Iowa_60m/"
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
        
    def set_masked_layer(self,year, outpath):
        sent_input = rio.open(self.inp_raster_path)
        cdl_layer = rio.open(self.CDL_path)
        # Masking of the CDL is done based on the availability of yield value for those counties at that year
        masked_gdf_path = self.outdir+"/yield_val/yield_"+str(year)+".shp"
        with fiona.open(masked_gdf_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        masked_cdl,masked_transform = mask.mask(cdl_layer, shapes, crop=True)
        
        out_meta = cdl_layer.meta
        out_meta.update({"driver": "GTiff",
                         "height": masked_cdl.shape[1],
                         "width": masked_cdl.shape[2],
                         "transform": masked_transform})
        
        with rio.open(self.CDL_dir+"CDL_Soybean_Iowa_60m_2021_clipped_masked.tif", "w", **out_meta) as dest:
            dest.write(masked_cdl)

        cdl_layer.close()
        
        cdl_mask_layer = rio.open(self.CDL_dir+"CDL_Soybean_Iowa_60m_2021_clipped_masked.tif")
        sent_input_12_bands = sent_input.read()[0:12]
        cdl = cdl_mask_layer.read(1)
        sent_masked = sent_input_12_bands*cdl
        
        sent_masked[sent_masked == 0] = np.nan
        
        sent_masked_norm = self.scaler.fit_transform(sent_masked.reshape(-1, sent_masked.shape[-1])).reshape(sent_masked.shape)
        
        meta = sent_input.meta.copy()
        meta.update(count=12)
        
        self.masked_out_file = outpath+"/sentinel_masked.tif"
    
        with rio.open(self.masked_out_file, 'w', **meta) as outds:
            outds.write(sent_masked_norm)
       
        show(sent_masked_norm[7],cmap="Greens")
        plt.savefig(os.path.join(self.plot_dir,"sentinel_masked.png"))
        plt.close()
        sent_input.close()
        cdl_mask_layer.close()
        
    def get_tile_patches(self, in_path, fname_prefix, patch_dim):

        # in_path = os.path.join(self.sentinel_image_dir,"sent2_2021_Iowa_60m")
        # in_path = glob.glob(in_path+"/*.tif")
        
        # in_path = [self.inp_raster_path,self.CDL_pathCDL]
        # for image in in_path:
        input_filename = in_path
        out_path = 'Input/sentinel/2021/sent2_2021_Iowa_60m/Iowa_masked_patches/'
        output_filename = fname_prefix+'_{}-{}.tif'
        print(input_filename)
        with rio.open(input_filename) as inds:
            meta = inds.meta.copy()
            for window, transform in self.get_tiles(inds,patch_dim):
                print(window)
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                # if fname_prefix== "cdl":
                #     outpath = os.path.join(out_path,output_filename.format(int(window.col_off/2), int(window.row_off/2)))
                # else:
                if np.isnan(inds.read(window=window)).all():
                    continue
                outpath = os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off)))
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))
              
    def get_ndvi_patch(self,patch_src):
        smin=0 
        smax=255

        x = patch_src.read(8) #NIR Band
        bandNIR = ( x - np.nanmin(x) ) * (smax - smin) / ( np.nanmax(x) - np.nanmin(x) ) + smin
        y = patch_src.read(4) #Red Band
        bandRed = ( y - np.nanmin(y) ) * (smax - smin) / ( np.nanmax(y) - np.nanmin(y) ) + smin
        ndvi = np.zeros(patch_src.read(1).shape, dtype=rio.float32)
        ndvi = ((bandNIR.astype(float)-bandRed.astype(float))/(bandNIR.astype(float)+bandRed.astype(float)))
        avg_ndvi = np.nanmean(ndvi)
        min_ndvi = np.nanmin(ndvi)
        max_ndvi = np.nanmax(ndvi)
        return avg_ndvi,min_ndvi,max_ndvi

    
    def set_target_for_patches(self,state_name):
        yield_inp_gdf = gpd.read_file(self.outdir+"/yield_val/yield_2021.shp")
        yield_inp = yield_inp_gdf[yield_inp_gdf['STATE_NAME'] == state_name]
        yield_inp = yield_inp.drop_duplicates(subset='County', keep="first") #Removing duplicates complicates since there are counties with same name in different states. Need to find a better solution to remove duplicates instead of using County column as subset
        #1 Bushel =  27.2 Kg
        # 1 acre is 4046.86 sq. m
        yield_kg_per_sq_m = 27.2/4046.86
        yield_inp["yield_in_kg_per_sqm"] = yield_inp["Value"]*yield_kg_per_sq_m
        
        patch_files_path = 'Input/sentinel/2021/sent2_2021_Iowa_60m/Iowa_masked_patches/'
        patch_files = glob.glob(patch_files_path+"*.tif")
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
            print(avg_ndvi,min_ndvi,max_ndvi)
            patch_bounds = list(patch_src.bounds)
            yield_inp_clip = gpd.clip(yield_inp,patch_bounds)
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
        target_dict["yld_kg_sqm"] = target_yield_list
        target_dict["ndvi_avg"] = ndvi_avg
        target_dict["ndvi_max"] = ndvi_max
        target_dict["ndvi_min"] = ndvi_min
        target_dict["geometry"] = patch_geom
        target_gdf = gpd.GeoDataFrame(target_dict,crs="EPSG:4269")
        out_path = os.path.join(self.sentinel_image_dir,"Target/Iowa_2021.shp")
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
        # self.set_masked_layer(2021,os.path.join(self.sentinel_image_dir,"sent2_2021_Iowa_60m/"))
        # self.masked_out_file = os.path.join(self.sentinel_image_dir,"sent2_2021_Iowa_60m/sentinel_masked.tif")
        # self.get_tile_patches(self.masked_out_file,"sentinel",self.patch_size)
        # self.get_tile_patches(self.CDL_path,"cdl",self.patch_size)
        self.set_target_for_patches("Iowa")
        
if __name__ == "__main__":
    
    dprep = DataPreparation()
    dprep.run()