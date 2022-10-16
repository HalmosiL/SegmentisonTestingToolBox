import os
import time
import logging
import argparse
import sys

import cv2
import numpy as np
import torch
import glob
from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

cv2.ocl.setUseOpenCL(False)

global detail
detail = True

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/conf.yaml', help='config file')
    parser.add_argument('--attack', action='store_true', help='evaluate the model with attack or not')
    parser.add_argument('opts', help='see config/conf.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None

    global attack_flag
    attack_flag = True

    cfg = config.load_cfg_from_cfg_file(args.config)
    
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def cal_acc(data_list, pred_folder, classes, names):
    global detail
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    len_ = len(glob.glob(pred_folder + "/" + "*"))

    for i, (image_path, target_path) in enumerate(data_list):        
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (1024, 512), interpolation=cv2.INTER_NEAREST)

        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        
        if(detail):
            print('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

        if(i + 1 == len_):
            break

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    
    if(detail):
        for i in range(classes):
            print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

def main(save_folder=None):
    global args, logger
    args = get_parser()
    logger = get_logger()
    
    if(save_folder is None):
        save_folder = args.save_folder

    gray_folder = os.path.join(save_folder + "/", 'gray')
    color_folder = os.path.join(save_folder + "/", 'color')

    test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)

    names = [line.rstrip('\n') for line in open(args.names_path)]

    index_start = args.index_start
    index_end = len(test_data.data_list)

    test_data.data_list = test_data.data_list[index_start:index_end]
    cal_acc(test_data.data_list, gray_folder, args.classes, names)


if __name__ == '__main__':        
    if(len(sys.argv) > 2):
        if(sys.argv[1] == "no"):
            detail = False
        
        for i in range(len(sys.argv) - 2):
            print(sys.argv[i + 2])
            main(sys.argv[i + 2])
            
            print()
    else:
        main()
