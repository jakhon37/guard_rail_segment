from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
import torch
import os
import pickle
import argparse
import json

setup_logger()

def _get_parsed_args() -> argparse.Namespace:
    """
    Create an argument parser and parse arguments.
    :return: parsed arguments as a Namespace object
    """
    parser = argparse.ArgumentParser(description="Detectron2 demo")
    parser.add_argument(
        "--config_file", # https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py
        default= "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",  #faster_rcnn_R_50_C4.yaml",  #Cityscapes/mask_rcnn_R_50_FPN.yaml",             
        help="Base model configuration and architecture file to be used for model training. "        # PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml
    )
    parser.add_argument(
        "--base_weight",       
        default= "", # "/home/jakhon37/PROJECTS/dttn2/output/output/model_final.pth", 
        help="Base model weight to be used for inferance. "   
    )   
    parser.add_argument(
        "--train_data",
        default= "./input/data_path.json", 
        help="Base model pickle file to be used for detection. "            
    )    
    parser.add_argument(
        "--num_class",
        default= 3, 
        help="Set number of classes in your data. "            
    )    
    parser.add_argument(
        "--batch_size",
        default= 1, 
        help="Set number of classes in your data. "            
    )
    parser.add_argument(
        "--num_workers",
        default= 2, 
        help="Set number of classes in your data. "            
    )
    parser.add_argument(
        "--num_iter",
        default= 500, 
        help="Set number of classes in your data. "            
    )
    parser.add_argument(
        "--lr_rate",
        default= 0.02 / 16 ,
        help="Set number of classes in your data. "            
    )
    parser.add_argument(
        "--eval_period",
        default= 0,
        help="Set number of classes in your data. "            
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device name (cpu/cudu)to be used for inferance. if not set - gpu"
    )    
    parser.add_argument(
        "--output",
        default="./output/seg_new",
        help="A file or directory to save output visualizations. "
        "If not given, will save on default location.",
    )
    parser.add_argument(
        "--task",
        default= "train", # "eval", "all",
        help="choose which task to process -> train / eval / train_eval"
    )    
    parser.add_argument(
        "--resume",
        default= False , # "eval",
        help="set training to resume or not True / False "
    ) 
    return parser.parse_args()


def get_train_cfg(config_file_path, num_classes, device,train_dataset_name, val_dataset_name, test_dataset,
                  output_dir, batch_size, num_workers, num_iter, lr_rate, eval_period):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # if train_resume == "True":
    #     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # else:
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file_path)
    cfg.DATASETS.TRAIN = (train_dataset_name,) # ("dataset_train")
    cfg.DATASETS.TEST =  (test_dataset,) #("test_train")
    cfg.DATASETS.VAL = (val_dataset_name,) # ("validation_train")
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = batch_size #1
    cfg.SOLVER.MAX_ITER = int(num_iter) #500
    cfg.SOLVER.BASE_LR = lr_rate # 0.02 * cfg.SOLVER.IMS_PER_BATCH  / 16  # pick a good LR
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg

def reg_dataset(train_data):
    with open(train_data, 'r') as file:
        line = json.load(file)    
    register_coco_instances(name=line["train_dataset_name"], metadata={}, json_file=line["train_label_path"], image_root=line["train_image_path"])
    register_coco_instances(name=line["validation_dataset_name"], metadata={}, json_file=line["validation_label_path"], image_root=line["validation_image_path"])
    register_coco_instances(name=line["test_dataset_name"], metadata={}, json_file=line["test_label_path"], image_root=line["test_image_path"])
    return line["train_dataset_name"], line["validation_dataset_name"], line["test_dataset_name"]

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if (args.output) is None:
        os.makedirs(args.output, exist_ok=True)
        output_folder = args.output
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

if __name__ == '__main__':
    args: argparse.Namespace = _get_parsed_args()
    train_dataset, validation_dataset, test_dataset = reg_dataset(args.train_data)
    print(f'Registering the data {train_dataset}')
    cfg = get_train_cfg(args.config_file, args.num_class, args.device, train_dataset, validation_dataset, test_dataset, args.output, args.batch_size, args.num_workers, args.num_iter, args.lr_rate, args.eval_period)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open((f'{args.output}/cfg_road.pickle'), "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
    if args.task == "train":
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()
    if args.task == "eval":
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator(validation_dataset, cfg, False, cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, validation_dataset)
        k = inference_on_dataset(predictor.model, val_loader, evaluator)
    if args.task == "train_evval":
        trainer = DefaultTrainer(cfg)
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator(validation_dataset, cfg, False, cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, validation_dataset)
        k = inference_on_dataset(predictor.model, val_loader, evaluator)
        
        
        