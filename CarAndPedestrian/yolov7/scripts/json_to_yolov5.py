from pathlib import Path 
import shutil
import random 
from tqdm import tqdm
import numpy as np
import json
import argparse
import yaml

LABELS = ['ConcreteCrack','Spalling','Efflorescene','Exposure','PaintDamage','SteelDefect']
LABELS = ['Efflorescene']

def make_dirs_yolo(path:Path):
    # Create folders
    # path = path/'YOLOv5'
    if path.exists():
        shutil.rmtree(path)  # delete dir
    for p in (
        path,
        path / 'train'/ 'labels', 
        path / 'train'/ 'images', 
        path / 'valid'/ 'labels', 
        path / 'valid'/ 'images', 
        ):
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

def convert_coco_json(label_list=[],output_dir='train', use_segments=False, cls91to80=False):
    coco80 = []
    LABEL_DIR = f'{output_dir}/labels'
    IMG_DIR = f'{output_dir}/images'


    # Import json
    for json_file in tqdm(label_list):#sorted(Path(input_dir+'/'+'labels').resolve().glob('*.json')):
        # fn = Path(save_dir) / 'labels' / json_file.stem.replace('instances_', '')  # folder name
        # fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}

        # Write labels file
        for x in data['annotations']:

            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']

            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            # Segments
            if use_segments:
                segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
                s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                
            input_img_path = json_file.parents[1] / 'images' / img['file_name']

            output_img_path = output_dir/ 'images' / img['file_name'] 
            shutil.copy(input_img_path,output_img_path)
            # Write
            if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                cls = coco80[x['category_id'] - 1] if cls91to80 else x['category_id'] - 1  # class
                #Get class name from annotation
                class_ = x['attributes']['class']
                #Get class index from class name check if class is in LABELS
                if class_ in LABELS:
                    class_idx = LABELS.index(class_)
                else:
                    continue
              

                line = cls, *(s if use_segments else box)  # cls, box or segments
                with open((f'{LABEL_DIR}/{json_file.stem}.txt'), 'a') as file:
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


#Define the function to parse the command line arguments
def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Convert COCO dataset to YOLOv5 format')
    parser.add_argument('--input_dir', dest='input_dir',
                        help='Input directory',
                        default='',
                        type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='Output directory',
                        default='YOLOv5',
                        type=str)

    parser.add_argument('--label', 
    dest='label',
    help='Label to convert',
    required=True,
    )
   
    return parser.parse_args()
#Define the main function
def main():

    #Parse the command line arguments
    args = parse_args()

    # Create path for output directory

    output_dir = Path(args.output_dir)/'YOLOv5'

    make_dirs_yolo(output_dir) # Create folders
    #label
    LABELS = [args.label]
    #Print labels
    print(f'Converting labels: {LABELS}')

    #Input labels directory
    input_labels_dir = Path(args.input_dir)
    #list of annotation files
    list_annotation = list(Path(input_labels_dir).iterdir())
    random.shuffle(list_annotation)

    #Split the dataset into train and validation
    train_split = int(len(list_annotation) * 0.2)
    #validation split
    val = list_annotation[:train_split]
    #train split
    train = list_annotation[train_split:]

    TRAN_DATASET_DIR = output_dir / 'train' # Path to the train dataset
    VAL_DATASET_DIR = output_dir / 'valid' # Path to the validation dataset



    #Convert train dataset
    convert_coco_json(label_list=train,output_dir=TRAN_DATASET_DIR)

    #Convert validation dataset
    convert_coco_json(label_list=val,output_dir=VAL_DATASET_DIR)

    #Create yolo config dictionary
    yolo_config = {
        'path':'/usr/src/app/workspace/YOLOv5',
        'train':'train/images',
        'val':'valid/images',
        'test':'',
        'nc':len(LABELS),
        'names':LABELS,
        # 'list':range(5)
    }

    with open(f'{output_dir}/dataset.yaml', 'w') as file:
        documents = yaml.dump(yolo_config, file)

#Run the main function
if __name__ == '__main__':
    
    main()
