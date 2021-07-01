# author: spoon@nvidia
# description: siim-kitti converter for tlt-deepstream

import pandas as pd
import numpy as np
import ast
import argparse
import os
import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import matplotlib
import shutil
from PIL import Image


def siim_resize_image(path, pixel):
    data_file = dicom.read_file(path)
    img = data_file.pixel_array
    
    original_height = img.shape[0]
    original_width = img.shape[1]
    
    # scale pixel to 0-255
    img = (img - np.min(img)) / np.max(img)
    img = (img * 255).astype(np.uint8)
    
    # resize
    img = cv2.resize(img, (pixel,pixel))
    img_array = np.array(img)
    
    x_scale_factor, y_scale_factor = pixel / original_width, pixel / original_height
    
    return img_array, x_scale_factor, y_scale_factor


def siim_image_path_parser(train_dir):
    img_dict = dict()
    for study in os.listdir(train_dir):
        for sub_study in os.listdir("{}/{}".format(train_dir, study)):
            for dcm in os.listdir("{}/{}/{}".format(train_dir, study, sub_study)):
                img_dict[dcm.replace(".dcm", "")] = "{}/{}/{}/{}".format(train_dir, study, sub_study, dcm)    
    
    return img_dict


def siim_bbox_parser(x, y, width, height):
    
    # bbox format for kitti: left-top corner & right-bottom corner
    # left-top corner coordinates: (xmin, ymax)
    # right-bottom corner coordinates: (xmax, ymin)
    
    # xmin
    bbox_left = x
    
    # ymax
    bbox_top = y + height
    
    # xmax
    bbox_right = x + width
    
    # ymin
    bbox_bottom = y
    
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def convert_kitti(data, train_dir, output_dir, pixel, with_opacity=True, img_idx_array=None, size=None):
    
    if os.path.exists(output_dir) & (img_idx_array == None):
        shutil.rmtree(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir,"formatted/labels"))
        os.makedirs(os.path.join(output_dir,"formatted/images"))
        os.makedirs(os.path.join(output_dir,"studies"))
        
    df = data.copy()
    
    df.id = data.id.apply(lambda x: x.replace("_image", ""))
    img_mapper = siim_image_path_parser(train_dir)
    df["img_path"] = df.id.apply(lambda x: img_mapper[x])
                
    if img_idx_array == None:
        img_idx_array = np.linspace(1, size, num=size, dtype=int)
        np.random.shuffle(img_idx_array)
        img_idx_array = img_idx_array.tolist()
    
                
    if with_opacity:
        
        for _, info in df.iterrows():

            obj = info["label"].split()
            number_of_obj = len(obj) // 6
            
            index, object_count = 0, 0
            
            current_data_index = str(img_idx_array.pop()).zfill(6)
            
            # image preprocessing
            img_array, x_scale_factor, y_scale_factor = siim_resize_image(info["img_path"], pixel)
            
            # save image    
            im = Image.fromarray(img_array)
            im.save("{}/formatted/images/{}.png".format(output_dir, current_data_index))

            with open("{}/formatted/labels/{}.txt".format(output_dir, current_data_index), "a") as file:
                while object_count < number_of_obj:
                    kitti_object = {"type": "", "truncated": "0", "occluded": "0", "alpha": "0",
                                    "bbox_left": "", "bbox_top": "", "bbox_right": "", "bbox_bottom": "",
                                    "dim-height": "0", "dim-width": "0", "dim-length": "0",
                                    "loc-X": "0", "loc-Y": "0", "loc-Z": "0", "rotation-y": "0"}

                    obj_class = obj[index:index + 6][0]
                    kitti_object["type"] = obj_class
                    
                    bbox = list(ast.literal_eval(info["boxes"])[object_count].values())

                    kitti_object["bbox_left"] = str(bbox[0] * x_scale_factor)
                    kitti_object["bbox_top"] = str(bbox[1] * y_scale_factor)
                    kitti_object["bbox_right"] = str(bbox[2] * x_scale_factor)
                    kitti_object["bbox_bottom"] = str(bbox[3] * y_scale_factor)
                    
                    index += 6
                    object_count += 1

                    # save label
                    file.write(" ".join(list(kitti_object.values())) + "\n")
                    
                    # save study
                    with open("{}/studies/{}.txt".format(output_dir, info["StudyInstanceUID"]), "a") as study_file:
                        study_file.write(info["id"] + "\n")
                                                                                                                                        
    else:
        
        for image, info in df.iterrows():
            
            current_data_index = str(img_idx_array.pop()).zfill(6)
            
            # image preprocessing
            img_array, x_scale_factor, y_scale_factor = siim_resize_image(info["img_path"], pixel)
            
            # save image    
            im = Image.fromarray(img_array)
            im.save("{}/formatted/images/{}.png".format(output_dir, current_data_index))
                        
            with open("{}/formatted/labels/{}.txt".format(output_dir, current_data_index), "a") as file:
                
                kitti_object = {"type": "none", "truncated": "0", "occluded": "0", "alpha": "0",
                                    "bbox_left": "0", "bbox_top": "1", "bbox_right": "1", "bbox_bottom": "0",
                                    "dim-height": "0", "dim-width": "0", "dim-length": "0",
                                    "loc-X": "0", "loc-Y": "0", "loc-Z": "0", "rotation-y": "0"}
                
                # save label
                file.write(" ".join(list(kitti_object.values())) + "\n")
                
                # save study
                with open("{}/studies/{}.txt".format(output_dir, info["StudyInstanceUID"]), "a") as study_file:

                    study_file.write(info["id"] + "\n")
                                        
                    
    return img_idx_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", "--label", help="file path of label csv", type=str)
    parser.add_argument("--t", "--train", help="file path of train csv", type=str)
    parser.add_argument("--kf", "--output_dir", help="file path for kitti output", type=str)
    parser.add_argument("--px", "--pixel", help="specify pixel for resized images", type=str)
    args = parser.parse_args()
    
    if not args.l:
        print("you must specify label path and training set path")
        exit()
        
    label_file = pd.read_csv(args.l)
    
    if not args.t:
        print("you must specify label path and training set path")
        exit()
    
    if not args.kf:
        args.kf = "{}/SIIM-FISABIO-RSNA".format(os.getcwd())
        
    if not args.px:
        args.px = 256
        
    no_opacity = label_file[label_file["boxes"].isnull() == True]
    yes_opacity = label_file[label_file["boxes"].isnull() == False]
    
    idx_array = convert_kitti(yes_opacity, train_dir=args.t, output_dir=args.kf, pixel=args.px, with_opacity=True, size=len(label_file))
    convert_kitti(no_opacity, train_dir=args.t, output_dir=args.kf, pixel=args.px, with_opacity=False, img_idx_array=idx_array)


if __name__ == "__main__":
    main()

