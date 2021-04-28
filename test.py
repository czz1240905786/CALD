import json
import pdb
# /data01/zyh/ALDataset/annotations/labeled_annotation.json

with open("/data01/zxl/KeyVehicleDetection/dataset/BITVehicle_Dataset/annotations/valid_annotation.json","r") as f:
# with open("/data01/zyh/ALDataset/annotations/train_annotation_new.json","r") as f:
    data = json.load(f)
    # print(data.keys())
    # pdb.set_trace()
    for annotation in data["annotations"]:
        annotation["category_id"] = annotation["category_id"] + 1
        # print(annotation)
    for category in data["categories"]:
        category["id"] = category["id"] + 1
    # pdb.set_trace()
    fp = open("/data01/zyh/ALDataset/annotations/valid_annotation_new.json","w")
    json.dump(data,fp)
    # pdb.set_trace()