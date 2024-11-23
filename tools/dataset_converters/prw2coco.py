import argparse
import os
import os.path as osp
from collections import defaultdict
from scipy.io import loadmat
import re

import mmengine
import numpy as np
from tqdm import tqdm

from mmengine.fileio import dump, list_from_file
from mmengine.utils import mkdir_or_exist, scandir, track_iter_progress
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PRW label and detections to COCO format.')
    parser.add_argument('-i', '--input', default='', type=str, help='The root path of PRW')
    parser.add_argument(
        '-o', '--output', type=str, default='', help='path to save coco formatted label file')
    return parser.parse_args()


def collect_image_infos(path, exclude_extensions=None):
    img_infos = []

    images_generator = scandir(path, recursive=True)
    for image_path in track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos

def get_cam_id(img_name):
    match = re.search(r"c\d", img_name).group().replace("c", "")
    return int(match)

def main():
    args = parse_args()
    if not osp.isdir(args.output):
        os.makedirs(args.output)
    img_id, ann_id = 0, 0
    ins_id = 1
    count_id = []
    
    
    # train
    imgs = loadmat(osp.join(args.input, "frame_train.mat"))["img_index_train"]
    img_list = [img[0][0] + ".jpg" for img in imgs]
    coco = dict()
    coco['images'] = []
    coco['categories'] = [{'id': 0, 'name': 'person'}]
    coco['annotations'] = []
    
    for img_name in img_list:
        
        ## Image
        image_path = osp.join(args.input, "frames", img_name)
        img_pillow = Image.open(image_path)
        image_item = dict() 
        image_item['id'] = int(img_id)
        image_item['file_name'] = str(image_path)
        image_item['height'] = int(img_pillow.height)
        image_item['width'] = int(img_pillow.width)
        coco['images'].append(image_item)
        
        ## annotations
        anno_path = osp.join(args.input, "annotations", img_name)
        anno = loadmat(anno_path)
        box_key = "box_new"
        if box_key not in anno.keys():
            box_key = "anno_file"
        if box_key not in anno.keys():
            box_key = "anno_previous"

        rois = anno[box_key][:, 1:]
        ids = anno[box_key][:, 0]
        rois = np.clip(rois, 0, None)
        assert len(rois) == len(ids)
        ids[ids == -2] = 5555
        
        for i in range(rois.shape[0]):
            if (ids[i] > ins_id) and (ids[i] != 5555):
                ins_id = ids[i]
            count_id.append(int(ids[i]))
            roi = rois[i]
            anno_item = dict()
            anno_item["id"] = ann_id
            anno_item["image_id"] = img_id
            anno_item["area"] = roi[2] * roi[3]
            anno_item["bbox"] = [roi[0], roi[1], roi[2], roi[3]]
            anno_item["category_id"] = 0
            anno_item["instance_id"] = int(ids[i])
            anno_item["cam_id"] = int(get_cam_id(img_name))
            anno_item["iscrowd"] = False
            anno_item["isquery"] = False
            ann_id += 1
            coco['annotations'].append(anno_item)
        img_id += 1
        
    print("train max_ids:", ins_id)
    print("train ann_ids:", ann_id)
    print(len(img_list), "images for train")
    # save_dir = os.path.join(args.input, 'annotations')
    # mkdir_or_exist(save_dir)
    save_path = os.path.join(args.output, "train_cocoformat.json")
    dump(coco, save_path)
    print(f'save json file: {save_path}')
    
    img_id, ann_id = 0, 0
    ins_id = 1
    
    ## val
    imgs = loadmat(osp.join(args.input, "frame_test.mat"))["img_index_test"]
    img_list = [img[0][0] + ".jpg" for img in imgs]
    coco = dict()
    coco['images'] = []
    coco['categories'] = [{'id': 0, 'name': 'person'}]
    coco['annotations'] = []
    query_info = osp.join(args.input, "query_info.txt")
    with open(query_info, "rb") as f:
        raw = f.readlines()
    
    
    queries = dict()
    for line in raw:
        linelist = str(line, "utf-8").split(" ")
        pid = int(linelist[0])
        x, y, w, h = (
            float(linelist[1]),
            float(linelist[2]),
            float(linelist[3]),
            float(linelist[4]),
        )
        roi = np.array([x, y, w, h])
        roi = np.clip(roi, 0, None)  # several coordinates are negative
        img_name = linelist[5][:-2] + ".jpg"
        if img_name in queries:
            queries[img_name]["boxes"] = np.concatenate((queries[img_name]["boxes"], roi[np.newaxis, :]), axis=0)
            queries[img_name]["pids"] = np.concatenate((queries[img_name]["pids"], np.array([pid])), axis=0)
            queries[img_name]["cam_id"] = np.concatenate((queries[img_name]["cam_id"], np.array([get_cam_id(img_name)])), axis=0)
        else:
            queries[img_name]={
                    # "img_path": osp.join(args.input, "frames", img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([pid]),
                    "cam_id": np.array([get_cam_id(img_name)]),
                }
    # queries = {}
    # for line in raw:
    #     linelist = str(line, "utf-8").split(" ")
    #     pid = int(linelist[0])
    #     x, y, w, h = (
    #         float(linelist[1]),
    #         float(linelist[2]),
    #         float(linelist[3]),
    #         float(linelist[4]),
    #     )
    #     roi = np.array([x, y, x + w, y + h]).astype(np.int32)
    #     roi = np.clip(roi, 0, None)  # several coordinates are negative
    #     img_name = linelist[5][:-2] + ".jpg"
    #     queries['bbox'] = 
    
    query_count = 0
    for img_name in img_list:
        ## Image
        # query = dict()
        # if img_name in queries:
            # print()
        #     query["boxes"] = queries[img_name]["boxes"]
        #     query["pids"] = queries[img_name]["pids"]
        #     query["cam_id"] = queries[img_name]["cam_id"]
        image_path = osp.join(args.input, "frames", img_name)
        img_pillow = Image.open(image_path)
        image_item = dict() 
        image_item['id'] = int(img_id)
        image_item['file_name'] = str(image_path)
        image_item['height'] = int(img_pillow.height)
        image_item['width'] = int(img_pillow.width)
        # image_item['query'] = query
        coco['images'].append(image_item)
        

        
        ## annotations
        anno_path = osp.join(args.input, "annotations", img_name)
        anno = loadmat(anno_path)
        box_key = "box_new"
        if box_key not in anno.keys():
            box_key = "anno_file"
        if box_key not in anno.keys():
            box_key = "anno_previous"

        rois = anno[box_key][:, 1:]
        ids = anno[box_key][:, 0]
        rois = np.clip(rois, 0, None)
        assert len(rois) == len(ids)
        ids[ids == -2] = 5555
        if img_name in queries:
            # print()
            bbox_list = queries[img_name]['boxes']
            pids_list = queries[img_name]['pids']
            cam_id_list = queries[img_name]['cam_id']
            count_gt = bbox_list.shape[0]
            count = 0
            for i in range(rois.shape[0]):
                if (ids[i] > ins_id) and (ids[i] != 5555):
                    ins_id = ids[i]
                roi = rois[i]
                anno_item = dict()
                anno_item["id"] = ann_id
                anno_item["image_id"] = img_id
                anno_item["area"] = roi[2] * roi[3]
                anno_item["bbox"] = [roi[0], roi[1], roi[2], roi[3]]
                anno_item["category_id"] = 0
                anno_item["instance_id"] = int(ids[i])
                anno_item["cam_id"] = int(get_cam_id(img_name))
                anno_item["iscrowd"] = False
                anno_item["isquery"] = False
                for j, gt_bbox in enumerate(bbox_list):
                    if abs((gt_bbox - roi)).sum()<= 0.001:
                        assert (pids_list[j] == int(ids[i])), "instance_ids not equall"
                        assert (cam_id_list[j] == anno_item["cam_id"]), "cam ids not same"
                        count +=1
                        anno_item["isquery"] = True
                        
                        break
                ann_id += 1
                coco['annotations'].append(anno_item)
            assert (count == count_gt), "not query"
            
        else:
            for i in range(rois.shape[0]):
                if (ids[i] > ins_id) and (ids[i] != 5555):
                    ins_id = ids[i]
                roi = rois[i]
                anno_item = dict()
                anno_item["id"] = ann_id
                anno_item["image_id"] = img_id
                anno_item["area"] = roi[2] * roi[3]
                anno_item["bbox"] = [roi[0], roi[1], roi[2], roi[3]]
                anno_item["category_id"] = 0
                anno_item["instance_id"] = int(ids[i])
                anno_item["cam_id"] = int(get_cam_id(img_name))
                anno_item["iscrowd"] = False
                anno_item["isquery"] = False
                ann_id += 1
                coco['annotations'].append(anno_item)
        img_id += 1
    # for an in coco['annotations']:
    #     if an["isquery"] == True:
    #         query_count+=1
    
    print("max_ids:", ins_id)
    print("test max_ids:", ins_id)
    print("test ann_ids:", ann_id)
    print(len(img_list), "images for tests")
    # save_dir = os.path.join(args.input, 'annotations')
    # mkdir_or_exist(save_dir)
    save_path = os.path.join(args.output, "test_cocoformat.json")
    dump(coco, save_path)
    print("query_count", query_count)
    print(f'save json file: {save_path}')
        
if __name__ == "__main__":
    main()
