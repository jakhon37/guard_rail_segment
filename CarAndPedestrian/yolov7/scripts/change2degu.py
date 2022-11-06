import os
from tqdm import tqdm
import json

root = "/media/HDD/ek_car_det/data/degu/"
all = os.listdir(root) 


json_paths = []
for _a in all:
    curr_path = os.path.join(root, _a)
    if os.path.isdir(curr_path):
        curr_paths = os.listdir(curr_path)
        #print(curr_paths)   
        if 'json' not in curr_paths[-1]:
    	    continue
        else:
            json_paths.append(os.path.join(root, _a))


uni_labels = {}
for p in tqdm(json_paths):
    print(p)
    l = os.listdir(p)
    for _l in l:
        f = os.path.join(p, _l)
        print(f)
        with open(f, 'r') as jf:
            labels = json.load(jf)
            #print(labels)

        for label in labels['annotations']:
        #print(label['label'])
            if label['label'] not in uni_labels:
                uni_labels[label['label']] = label['label']
    print(uni_labels.keys())
print("done")
