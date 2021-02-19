# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

register_coco_instances('balloon_train', {},
                        'train.json',
                       '/data/balloon/train/')
register_coco_instances('balloon_val', {},
                        'val.json',
                       '/data/balloon/val/')

coco_val_metadata = MetadataCatalog.get("balloon_val")
coco_train_metadata = MetadataCatalog.get("balloon_train")
# dataset_dicts = DatasetCatalog.get("balloon_val")
dataset_dicts = DatasetCatalog.get("balloon_train")

for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=coco_val_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("img", vis.get_image()[:,:,::-1])
    cv2.waitKey(0)
