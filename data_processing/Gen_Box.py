import mrcfile
import numpy as np
import os
from data_processing.map_utils import find_top_density,permute_ns_coord_to_pdb, permute_map_coord_to_pdb, read_map_data, write_voxel_map
# from progress.bar import Bar

import torch
def gen_input_box(map_data, mapc, mapr, maps, new_origin, map_path, \
    box_size,stride,contour,train_save_path, visual_check=0):
    scan_x, scan_y, scan_z = map_data.shape
    count_voxel = 0
    count_iter=0
    Coord_Voxel = []
    # bar = Bar('Preparing Input: ', max=int(np.ceil(scan_x/stride)*np.ceil(scan_y/stride)*np.ceil(scan_z/stride)))

    box_list = []
    for x in range(0, scan_x, stride):
        x_end = min(x + box_size, scan_x)
        for y in range(0, scan_y, stride):
            y_end = min(y + box_size, scan_y)
            for z in range(0, scan_z, stride):
                count_iter+=1
                # bar.next()
                #print("1st stage: %.4f percent scanning finished"%(count_iter*100/(scan_x*scan_y*scan_z/(stride**3))),"location %d %d %d"%(x,y,z))
                z_end = min(z + box_size, scan_z)
                if x_end < scan_x:
                    x_start = x
                else:
                    x_start = x_end - box_size

                    if x_start<0:
                        x_start=0
                if y_end < scan_y:
                    y_start = y
                else:
                    y_start = y_end - box_size

                    if y_start<0:
                        y_start=0
                if z_end < scan_z:
                    z_start = z
                else:
                    z_start = z_end - box_size

                    if z_start<0:
                        z_start=0
                
                box_list.append((x_start, x_end, y_start, y_end, z_start, z_end))
    
    meaningful_indices = []
    input_list = []
    for idx, (x_start, x_end, y_start, y_end, z_start, z_end) in enumerate(box_list):
        #already normalized
        segment_map_voxel = np.zeros([box_size,box_size,box_size], dtype=np.float32)
        segment_map_voxel[:x_end-x_start,:y_end-y_start,:z_end-z_start]=map_data[x_start:x_end, y_start:y_end, z_start:z_end]
        input_list.append(segment_map_voxel)
        if contour <= 0:
            meaningful_density_count = np.count_nonzero(segment_map_voxel > 0)
        else:
            meaningful_density_count = np.count_nonzero(segment_map_voxel > contour)

        meaningful_density_ratio = meaningful_density_count / float(box_size ** 3)
        if meaningful_density_ratio > 0.001:
            print(f"box_list_idx: {idx}, location: {x_start} {x_end} {y_start} {y_end} {z_start} {z_end}")
            meaningful_indices.append(idx)
    return input_list, box_list, meaningful_indices

def generate_data_pair(
    input_map_path,
    input_cif_file,
    dssp_cif_file,
    save_input_dir,
    contour,
    box_size,
    stride,
    save_voxel_path,
    visual_check=0,
    log_path=None,
):
    os.makedirs(save_input_dir,exist_ok=True)
    with mrcfile.open(input_map_path, permissive=True) as map_mrc:
         #normalize data
        map_data = np.array(map_mrc.data)
        if not np.isfinite(map_data).all() or not np.isfinite(contour):
            raise ValueError("Map densities and contour must be finite before normalization")
        # get the value serve as 1 in normalization
        map_data[map_data < 0] = 0
        print("map density range: %f %f"%(0,np.max(map_data)))
        percentile_98 = find_top_density(map_data,0.98)

        print("map hist log percentage 98: ",percentile_98)
        map_data[map_data > percentile_98] = percentile_98
        min_value = np.min(map_data)
        max_value = np.max(map_data)
        if max_value <= min_value:
            raise ValueError("Cannot normalize a constant or empty-density map")
        map_data = (map_data-min_value)/(max_value-min_value)
        nxstart, nystart, nzstart = map_mrc.header.nxstart, \
                                    map_mrc.header.nystart, \
                                    map_mrc.header.nzstart
        orig = map_mrc.header.origin
        orig = str(orig)
        orig = orig.replace("(", "")
        orig = orig.replace(")", "")
        orig = orig.split(",")
        nstart = [nxstart, nystart, nzstart]
        mapc = map_mrc.header.mapc
        mapr = map_mrc.header.mapr
        maps = map_mrc.header.maps
        print("detected mode mapc %d, mapr %d, maps %d" % (mapc, mapr, maps))
        nstart = permute_ns_coord_to_pdb(nstart, mapc, mapr, maps)
        new_origin = []
        for k in range(3):
            new_origin.append(float(orig[k]) + float(nstart[k]))

        print("Origin:", new_origin)

        print("given contour %f"%contour)
        if contour > 0:
            contour = (contour - min_value) / (max_value - min_value)
        print("revised contour %f"%contour)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{save_voxel_path} : {contour}\n")
        with mrcfile.new(os.path.join(save_input_dir, f"{save_voxel_path}_normal.mrc"), overwrite=True) as mrc:
            mrc.set_data(map_data.astype(np.float32))  # 注意保存为 float32 格式
            mrc.header.origin = map_mrc.header.origin
            mrc.voxel_size = map_mrc.voxel_size  # 可选：保留原始 voxel size
            mrc.header.nxstart = nxstart
            mrc.header.nystart = nystart
            mrc.header.nzstart = nzstart
            mrc.header.mapc = mapc
            mrc.header.mapr = mapr
            mrc.header.maps = maps
            mrc.update_header_from_data()
            
        box_dir = os.path.join(save_input_dir, f"{save_voxel_path}")
        input_list, box_list, meaningful_indices = gen_input_box(map_data,mapc, mapr, maps, new_origin, input_map_path, box_size,stride,contour,box_dir, visual_check=visual_check)
        
        return input_list, box_list, meaningful_indices
