import os
import time
import logging
import argparse
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn

from Adversarial import Cosine_PDG_Adam, model_immer_attack_auto_loss
from model.pspnet import PSPNet_DDCAT, DeepLabV3_DDCAT, PSPNet, DeepLabV3, Dummy
from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

cv2.ocl.setUseOpenCL(False)

MODEL_SLICE = None

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
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
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


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.test_h - 1) % 8 == 0 and (args.test_w - 1) % 8 == 0
    assert args.split in ['train', 'val', 'test']
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    global mean_origin
    global std_origin
    mean_origin = [0.485, 0.456, 0.406]
    std_origin = [0.229, 0.224, 0.225]

    args.save_folder = args.save_folder + os.environ['model'] + '_cosine_' + os.environ['epsilon'] + "/"

    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    index_start = args.index_start

    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))

    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]

    global MODEL_SLICE

    DEV = True

    if not args.has_prediction:
        if not DEV:
            if(os.environ['model'] == "ddcat"):
                model = PSPNet_DDCAT(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
            elif(os.environ['model'] == "normal" or os.environ['model'] == "sat"):
                model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
        else:
            model = Dummy()

        logger.info(model)
        print("cuda:" + str(args.test_gpu[0]))

        model = model.to("cuda:" + str(args.test_gpu[0]))
        MODEL_SLICE = model.getSliceModel().to("cuda:" + str(args.test_gpu[0]))

        logger.info("Load Models")

        model = torch.nn.DataParallel(model, device_ids=[args.test_gpu[0]]).to("cuda:" + str(args.test_gpu[0]))

        cudnn.benchmark = True
        
        if(not DEV):
            if os.path.isfile(args.model_path):
                logger.info("=> loading checkpoint '{}'".format(args.model_path))

                if(os.environ['model'] == "ddcat"):
                    checkpoint = torch.load(args.model_path_ddcat, map_location="cuda:" + str(args.test_gpu[0]))
                elif(os.environ['model'] == "normal"):
                    checkpoint = torch.load(args.model_path_normal, map_location="cuda:" + str(args.test_gpu[0]))
                elif(os.environ['model'] == "sat"):
                    checkpoint = torch.load(args.model_path_sat, map_location="cuda:" + str(args.test_gpu[0]))
                
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                logger.info("=> loaded checkpoint '{}'".format(args.model_path))
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

        logger.info("Start Testing")
        test(test_loader, test_data.data_list, model, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, gray_folder, color_folder, colors)
    if args.split != 'test':
        cal_acc(test_data.data_list, gray_folder, args.classes, names)


def net_process(model, image, target, mean, std=None):
    mean_origin = [0.485, 0.456, 0.406]
    std_origin = [0.229, 0.224, 0.225]

    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    target = torch.from_numpy(target).long()

    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)

    input = input.unsqueeze(0).to(args.test_gpu[0])
    target = target.unsqueeze(0).to(args.test_gpu[0])

    attack = Cosine_PDG_Adam(
        step_size=float(os.environ['alpha']),
        clip_size=float(os.environ['epsilon'])
    )
    
    adver_input = model_immer_attack_auto_loss(input, MODEL_SLICE, attack, args.number_of_steps, args.test_gpu[0])

    with torch.no_grad():
        output = model(adver_input)

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)

    output = F.softmax(output, dim=1)
    
    output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)

    adver_input[:, 0, :, :] = adver_input[:, 0, :, :] * std_origin[0] + mean_origin[0]
    adver_input[:, 1, :, :] = adver_input[:, 1, :, :] * std_origin[1] + mean_origin[1]
    adver_input[:, 2, :, :] = adver_input[:, 2, :, :] * std_origin[2] + mean_origin[2]

    adver_input = adver_input[0].cpu().detach().numpy().transpose((1, 2, 0))

    return output, adver_input


def scale_process(model, image, target, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
        target = cv2.copyMakeBorder(target, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)

    adv_image_full = np.zeros((new_h, new_w, 3), dtype=float)

    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)

    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()

            target_crop = target[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            pred, adv = net_process(model, image_crop, target_crop, mean, std)

            adv_image_full[s_h:e_h, s_w:e_w, :] = adv
            prediction_crop[s_h:e_h, s_w:e_w, :] += pred

    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction, adv_image_full

def test(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        color_path = os.path.join(color_folder, image_name + '.png')

        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        target = np.squeeze(target.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)

        h, w, _ = image.shape

        prediction = np.zeros((h, w, classes), dtype=float)
        adv_full = np.zeros((h, w, 3), dtype=float)

        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)

            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            target_scale = cv2.resize(target.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            pred, adv = scale_process(model, image_scale, target_scale, classes, crop_h, crop_w, h, w, mean, std)
            prediction += pred
            adv_full = adv
        
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)

        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)

        adv_full_copy =  np.copy(adv_full)
        adv_full[:, :, 0] =  adv_full_copy[:, :, 2]
        adv_full[:, :, 2] =  adv_full_copy[:, :, 0]

        check_makedirs(color_folder[:-1 * len("color")] + "adv_images")

        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')

        cv2.imwrite(color_folder[:-1 * len("color")] + "adv_images/" + image_name + '.png', adv_full * 255)
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

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
        logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        with open('Failed.txt', 'w') as file:
            file.write(str(e))
