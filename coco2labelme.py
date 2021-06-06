'''
Usage:
1. put the file in picture folder. Eg. train/coco2labelme.py
2. run the code. python coco2labelme.py --cocoanno coco_annotations_file_path
'''
import base64
import labelme
import json
import argparse
import pdb
from pycocotools.coco import COCO
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cocoanno', help='coco annotations/CALD generate annotations')
# parser.add_argument('--labelmeanno', help='labelme annotations/output annotations')
args = parser.parse_args()

# coco_data=json.load(open(args.cocoanno,"r"))
coco=COCO(args.cocoanno)
imgIds = coco.getImgIds()
# [{'id': 165, 'image_id': 153, 'category_id': 2, 'bbox': [627, 5, 305, 371], 'area': 113155, 'iscrowd': 0}, {'id': 166, 'image_id': 153, 'category_id': 4, 'bbox': [1003, 384, 424, 521], 'area': 220904, 'iscrowd': 0}]


for img_index in tqdm(imgIds):
    shapes_list=[]
    annIds=coco.getAnnIds(img_index)
    anns=coco.loadAnns(annIds)
    for anno in anns:
        label=coco.loadCats(anno["category_id"])[0]["name"]
        points=[[anno["bbox"][0],anno["bbox"][1]],[anno["bbox"][0]+anno["bbox"][2],anno["bbox"][1]+anno["bbox"][3]]]
        shapes_list.append({
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {}})
    loadimages=coco.loadImgs(img_index)
    imagePath = loadimages[0]["file_name"]
    imageData=base64.b64encode(labelme.LabelFile.load_image_file(imagePath)).decode('utf-8')
    imageHeight=loadimages[0]["height"]
    imageWidth=loadimages[0]["width"]
    labelme_data={
        "version": "4.2.10",
        "flags": {},
        "shapes": shapes_list,
        "imagePath": imagePath,
        "imageData": imageData,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth
    }
    json.dump(labelme_data,open(imagePath[:-4]+".json","w"))
    # break
print("finish!")
