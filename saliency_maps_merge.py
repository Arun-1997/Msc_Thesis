from rasterio.merge import merge
import rasterio as rio
import glob


def merge_saliency_maps(year):
    print(year)
    inp_file_train = glob.glob("Output/saliency_maps/gradCAM_nomask_sent/train/*_"+str(year)+"_*.tif")
    inp_file_test = glob.glob("Output/saliency_maps/gradCAM_nomask_sent/test/*_"+str(year)+"_*.tif")
    inp_file_list = inp_file_train+inp_file_test
    inp_files_layer = [rio.open(i) for i in inp_file_list]

    merged_layer = merge(inp_files_layer)
    meta = inp_files_layer[0].meta.copy()
    meta.update(count=15)
    meta.update({"driver": "GTiff",
                 "height": merged_layer[0].shape[1],
                 "width": merged_layer[0].shape[2]})
    
    for i in inp_files_layer:
        i.close()

    merged_out_file = "Output/saliency_maps/gradCAM_nomask_sent/gradCAM_nomask_merged_"+str(year)+".tif"
    with rio.open(merged_out_file, 'w', **meta) as outds:
        outds.write(merged_layer[0])

        

merge_saliency_maps(2017)
merge_saliency_maps(2018)
merge_saliency_maps(2019)
merge_saliency_maps(2020)
merge_saliency_maps(2021)