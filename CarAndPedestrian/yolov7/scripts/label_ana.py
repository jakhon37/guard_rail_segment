import os
import json
from tqdm import tqdm



root = "/media/HDD/ek_car_det/data/seoul/"
paths = ["20201125_seoul_front_5_0006/",
        "20201125_seoul_front_5_0007",
        "20201125_seoul_front_5_0008",
        "20201224_seoul_front_5_0012",
        "20201224_seoul_front_5_0013",
        "20201224_seoul_front_5_0014"
        ]
root = "/media/HDD/ek_car_det/data/gyongido/"
paths = [
        "20201125_gyongido_front_5_0006",
        "20201125_gyongido_front_5_0007",
        "20201125_gyongido_front_5_0008",
        "20201126_gyongido_front_5_0001",
        "20201126_gyongido_front_5_0002",
        "20201126_gyongido_front_5_0004",
        "20201127_gyongido_front_5_0003",
        "20201217_gyongido_front_4_0084",
        "20201217_gyongido_front_4_0085",
        "20201217_gyongido_front_4_0086",
        "20201217_gyongido_front_4_0087",
        "20201217_gyongido_front_4_0088",
        "20201217_gyongido_front_4_0089",
        "20201217_gyongido_front_4_0090",
        "20201222_gyongido_front_5_0009",
        "20201222_gyongido_front_5_0010",
        "20201222_gyongido_front_5_0011",
        "20201224_gyongido_front_5_0012",
        "20201224_gyongido_front_5_0013",
        "20201224_gyongido_front_5_0014",
        ]
print(root)
uni_labels = {}
for p in tqdm(paths):
    l = os.listdir(root + p)
    for _l in l:
        f = os.path.join(root, p, _l)
        with open(f, 'r') as jf:
            labels = json.load(jf)
            #print(labels)
    
        for label in labels['annotations']:
        #print(label['label'])
            if label['label'] not in uni_labels:
                uni_labels[label['label']] = label['label']
    print(uni_labels.keys())
print("done")


