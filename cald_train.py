import datetime
import os
import time
import random
import math
import sys
import numpy as np
import math
import scipy.stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from detection.coco_utils import get_coco, get_coco_kp
from detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from detection.engine import coco_evaluate, voc_evaluate
from detection import utils
from detection import transforms as T
from detection.transforms import xy2wh
from detection.train import *
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
from cald.cald_helper import *
from ll4al.data.sampler import SubsetSequentialSampler
from detection.frcnn_la import fasterrcnn_resnet50_fpn_feature
from detection.retinanet_cal import retinanet_mobilenet, retinanet_resnet50_fpn_cal
import pdb
import matplotlib
from tqdm import tqdm
import os
import shutil
import json
import mmcv
matplotlib.rcParams['font.sans-serif']=['SimHei']   
matplotlib.rcParams['axes.unicode_minus']=False     

def train_one_epoch(task_model, task_optimizer, data_loader, device, cycle, epoch, print_freq):
    # 设置self.trainning为True
    task_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('task_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Cycle:[{}] Epoch: [{}]'.format(cycle, epoch)

    task_lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        task_lr_scheduler = utils.warmup_lr_scheduler(task_optimizer, warmup_iters, warmup_factor)

    for images, targets, paths in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        task_loss_dict = task_model(images, targets)
        task_losses = sum(loss for loss in task_loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        task_loss_dict_reduced = utils.reduce_dict(task_loss_dict)
        task_losses_reduced = sum(loss.cpu() for loss in task_loss_dict_reduced.values())
        task_loss_value = task_losses_reduced.item()
        if not math.isfinite(task_loss_value):
            print("Loss is {}, stopping training".format(task_loss_value))
            print(task_loss_dict_reduced)
            sys.exit(1)

        task_optimizer.zero_grad()
        task_losses.backward()
        task_optimizer.step()
        if task_lr_scheduler is not None:
            task_lr_scheduler.step()
        metric_logger.update(task_loss=task_losses_reduced)
        metric_logger.update(task_lr=task_optimizer.param_groups[0]["lr"])
    return metric_logger


def calcu_iou(A, B):
    '''
    calculate two box's iou
    '''
    width = min(A[2], B[2]) - max(A[0], B[0]) + 1
    height = min(A[3], B[3]) - max(A[1], B[1]) + 1
    if width <= 0 or height <= 0:
        return 0
    Aarea = (A[2] - A[0]) * (A[3] - A[1] + 1)
    Barea = (B[2] - B[0]) * (B[3] - B[1] + 1)
    iner_area = width * height
    return iner_area / (Aarea + Barea - iner_area)


def get_uncertainty(task_model, unlabeled_loader, augs, num_cls, args):
    '''
    Arguments:
        task_model (nn.Module): model for evaluation
        unlabeled_loader : dataloader for unlabeled_pool
        augs (list[Str]): DataAugmentation methods
        num_cls (int): number of classes

    Returns:
        consistency_all (list[float]) : consistency for all unlabeled images      [N,]
        cls_all (list[list]) : class distribution for all unlabeled images        [(N,num_cls)]
        outputs_all (list[list[ndarray]]): predictions('boxes','labeles','scores') for all unlabeled images [(N,...)]
             'boxes' : (100,4)
             'labeles' : (100,)
             'scores' : (100,)
    '''
    for aug in augs:
        if aug not in ['flip', 'multi_ga', 'color_adjust', 'color_swap', 'multi_color_adjust', 'multi_sp', 'cut_out',
                       'multi_cut_out', 'multi_resize', 'larger_resize', 'smaller_resize', 'rotation', 'ga', 'sp']:
            print('{} is not in the pre-set augmentations!'.format(aug))
    task_model.eval()
    with torch.no_grad():
        consistency_all = []
        mean_all = []
        #  TODO: second stage Metric
        cls_all = []
        outputs_all = []
        
        for images, _, _ in tqdm(unlabeled_loader):
            
            torch.cuda.synchronize()
            # only support 1 batch size
            aug_images = []
            aug_boxes = []
            # batch_size=1 原则上不需要写for
            for image in images:
                # TODO: 🔖 A(M(x))
                output = task_model([F.to_tensor(image).cuda()])
                # 目前一张图片只会有一个标?
                outputs_all.append([output[0]['boxes'].cpu().numpy(),output[0]['labels'].cpu().numpy(),output[0]['scores'].cpu().numpy()])
                # output = task_model([image.cuda()])
                # 这里首先提取出image图片的模型运算结�?
                # pdb.set_trace()
                ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = output[0]['boxes'], output[0][
                    'prob_max'], output[0]['scores_cls'], output[0]['labels'], output[0]['scores']
                if len(ref_scores) > 40:
                    # 选择得分最高的50个框
                    inds = np.round(np.linspace(0, len(ref_scores) - 1, 50)).astype(int)
                    ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = ref_boxes[inds], prob_max[
                        inds], ref_scores_cls[inds], ref_labels[inds], ref_scores[inds]
                cls_corr = [0] * (num_cls)
                # cls_corr = [0] * (num_cls - 1)
                # 得到每个每个类对应的最大得�?
                for s, l in zip(ref_scores, ref_labels):
                    cls_corr[l-1] = max(cls_corr[l-1], s.item())
                cls_corrs = [cls_corr]
                if output[0]['boxes'].shape[0] == 0:
                    consistency_all.append(0.0)
                    cls_all.append(np.mean(cls_corrs, axis=0))
                    break
                # start augment
                if 'flip' in augs:
                    flip_image, flip_boxes = HorizontalFlip(image, ref_boxes)
                    aug_images.append(flip_image.cuda())
                    aug_boxes.append(flip_boxes.cuda())
                if 'ga' in augs:
                    ga_image = GaussianNoise(image, 16)
                    aug_images.append(ga_image.cuda())
                    aug_boxes.append(ref_boxes.cuda())
                if 'multi_ga' in augs:
                    for i in range(1, 7):
                        ga_image = GaussianNoise(image, i * 8)
                        aug_images.append(ga_image.cuda())
                        aug_boxes.append(ref_boxes.cuda())
                if 'color_adjust' in augs:
                    color_adjust_image = ColorAdjust(image, 1.5)
                    aug_images.append(color_adjust_image.cuda())
                    aug_boxes.append(ref_boxes)
                if 'color_swap' in augs:
                    color_swap_image = ColorSwap(image)
                    aug_images.append(color_swap_image.cuda())
                    aug_boxes.append(ref_boxes)
                if 'multi_color_adjust' in augs:
                    for i in range(2, 6):
                        color_adjust_image = ColorAdjust(image, i)
                        aug_images.append(color_adjust_image.cuda())
                        aug_boxes.append(ref_boxes)
                if 'sp' in augs:
                    sp_image = SaltPepperNoise(image, 0.1)
                    aug_images.append(sp_image.cuda())
                    aug_boxes.append(ref_boxes)
                if 'multi_sp' in augs:
                    for i in range(1, 7):
                        sp_image = SaltPepperNoise(image, i * 0.05)
                        aug_images.append(sp_image.cuda())
                        aug_boxes.append(ref_boxes)
                if 'cut_out' in augs:
                    cutout_image = cutout(image, ref_boxes, ref_labels, 2)
                    aug_images.append(cutout_image.cuda())
                    aug_boxes.append(ref_boxes)
                if 'multi_cut_out' in augs:
                    for i in range(1, 5):
                        cutout_image = cutout(image, ref_boxes, ref_labels, i)
                        aug_images.append(cutout_image.cuda())
                        aug_boxes.append(ref_boxes)
                if 'multi_resize' in augs:
                    for i in range(7, 10):
                        resize_image, resize_boxes = resize(image, ref_boxes, i * 0.1)
                        aug_images.append(resize_image.cuda())
                        aug_boxes.append(resize_boxes)
                if 'larger_resize' in augs:
                    resize_image, resize_boxes = resize(image, ref_boxes, 1.2)
                    aug_images.append(resize_image.cuda())
                    aug_boxes.append(resize_boxes)
                if 'smaller_resize' in augs:
                    resize_image, resize_boxes = resize(image, ref_boxes, 0.8)
                    aug_images.append(resize_image.cuda())
                    aug_boxes.append(resize_boxes)
                if 'rotation' in augs:
                    rot_image, rot_boxes = rotate(image, ref_boxes, 5)
                    aug_images.append(rot_image.cuda())
                    aug_boxes.append(rot_boxes)
                # TODO: 🔖 M(A(x))
                outputs = []
                for aug_image in aug_images:
                    # 虽然原图只有一张，但是aug_image�?�?
                    outputs.append(task_model([aug_image])[0])
                consistency_aug = []
                mean_aug = []
                for output, aug_box, aug_image in zip(outputs, aug_boxes, aug_images):
                    consistency_img = 1.0
                    mean_img = []
                    boxes, scores_cls, pm, labels, scores = output['boxes'], output['scores_cls'], output['prob_max'], \
                                                            output['labels'], output['scores']
                    cls_corr = [0] * (num_cls)
                    # cls_corr = [0] * (num_cls - 1)
                    for s, l in zip(scores, labels):
                        cls_corr[l-1] = max(cls_corr[l-1], s.item())
                    cls_corrs.append(cls_corr)
                    if len(boxes) == 0:
                        consistency_aug.append(0.0)
                        mean_aug.append(0.0)
                        continue
                    for ab, ref_score_cls, ref_pm, ref_score in zip(aug_box, ref_scores_cls, prob_max, ref_scores):
                        width = torch.min(ab[2], boxes[:, 2]) - torch.max(ab[0], boxes[:, 0])
                        height = torch.min(ab[3], boxes[:, 3]) - torch.max(ab[1], boxes[:, 1])
                        Aarea = (ab[2] - ab[0]) * (ab[3] - ab[1])
                        Barea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        iner_area = width * height
                        iou = iner_area / (Aarea + Barea - iner_area)
                        iou[width < 0] = 0.0
                        iou[height < 0] = 0.0
                        p = ref_score_cls.cpu().numpy()
                        q = scores_cls[torch.argmax(iou)].cpu().numpy()
                        m = (p + q) / 2
                        # TODO: 计算JS散度
                        js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
                        if js < 0:
                            js = 0
                        # consistency_img.append(torch.abs(
                        #     torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)]) - args.bp).item())
                        # TODO: 计算Metric
                        consistency_img = min(consistency_img, torch.abs(
                            torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)]) - args.bp).item())
                        mean_img.append(torch.abs(
                            torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)])).item())
                    consistency_aug.append(np.mean(consistency_img))
                    mean_aug.append(np.mean(mean_img))
                # 对于数据增强做平�?
                consistency_all.append(np.mean(consistency_aug))
                mean_all.append(mean_aug)
                # 类别分布做平�?
                cls_corrs = np.mean(np.array(cls_corrs), axis=0)
                # cls_corrs的维度为(5,6)�?是因�?1原图+4增强 �?是因为六个类�?
                cls_all.append(cls_corrs)
    mean_aug = np.mean(mean_all, axis=0)
    print(mean_aug)
    # consistency_all(num_img) cls_all(num_img,6)
    return consistency_all, cls_all, outputs_all


def cls_kldiv(labeled_loader, cls_corrs, budget, cycle):
    cls_inds = []
    result = []
    # pdb.set_trace()
    # 计算labeled pool中的概率分布
    for _, targets, _ in labeled_loader:
        for target in targets:
            cls_corr = [0] * cls_corrs[0].shape[0]
            
            # try:
            #     cls_corr = [0] * cls_corrs[0].shape[0]
            # except:
            #     pdb.set_trace()
            # 这是class的数�?
            for l in target['labels']:
                cls_corr[l-1] += 1
            result.append(cls_corr)
        # with open("vis/mutual_cald_label_{}_{}_{}_{}.txt".format(args.uniform, args.model, args.dataset, cycle),
        #           "wb") as fp:  # Pickling
        # pickle.dump(result, fp)
    for a in list(np.where(np.sum(cls_corrs, axis=1) == 0)[0]):
        cls_inds.append(a)
        # pdb.set_trace()
        # result.append(cls_corrs[a])
    while len(cls_inds) < budget:
        # batch cls_corrs together to accelerate calculating
        KLDivLoss = nn.KLDivLoss(reduction='none')
        _cls_corrs = torch.tensor(cls_corrs)
        # result按照输入图像的数量做一个均�?
        _result = torch.tensor(np.mean(np.array(result), axis=0)).unsqueeze(0)
        if args.uniform:
            p = torch.nn.functional.softmax(_result + _cls_corrs, -1)
            q = torch.nn.functional.softmax(torch.ones(_result.shape) / len(_result), -1)
            log_mean = ((p + q) / 2).log()
            jsdiv = torch.sum(KLDivLoss(log_mean, p), dim=1) / 2 + torch.sum(KLDivLoss(log_mean, q), dim=1) / 2
            jsdiv[cls_inds] = 100
            max_ind = torch.argmin(jsdiv).item()
            cls_inds.append(max_ind)
        else:
            p = torch.nn.functional.softmax(_result, -1)
            q = torch.nn.functional.softmax(_cls_corrs, -1)
            log_mean = ((p + q) / 2).log()
            jsdiv = torch.sum(KLDivLoss(log_mean, p), dim=1) / 2 + torch.sum(KLDivLoss(log_mean, q), dim=1) / 2
            jsdiv[cls_inds] = -1
            max_ind = torch.argmax(jsdiv).item() # 选取argmax最?
            cls_inds.append(max_ind)
        # result.append(cls_corrs[max_ind])
    return cls_inds

def save2file(dataloader,file : str = "/data01/gpl/ALDataset/BITVehicle_Dataset/annotations/train_annotation_gpl.json",
              img_id: int = None, anno_id: int = None, bboxes: np.ndarray = None, scores : np.ndarray = None,
              labeles: np.ndarray = None, trick: str = "threshold", output_all : list = None,cycle = 0,args=None):
    """
    Arguments:
        file (str): file to be saved
        dataloader (torch.utils.data.Dataloader): dataloader returns img,target,path
        img_id (int): max img_id in labeled_pool
        anno_id (int): max anno_id in labeled_pool
        bboxes (np.ndarray): bounding boxs to be saved                       [N,100,4]
        scores (np.ndarray): scores to be saved                              [N,100,]
        labeles (np.ndarray): labels to be saved                             [N,100,]
        trick (str): tricks for selecting bbox,scores,labeles
        output_all (list): output of all valid images which contain bboxes,labeles,scores [N,3,(100,*)]
            bboxes:  output_all[:,0,100,4]
            labeles: output_all[:,1,100, ]
            scores:  output_all[:,2,100, ]
            len(output_all)=60 len(output_all[0])=3
            后面的维数为ndarray
    Returns:
        None
    """
    label_all_=[]
    try:
        os.mkdir('./result_image/cycle'+str(cycle))
    except:
        print('cycle'+str(cycle)+' already exist!')
    score_all_ = []
    scores_ = []
    if 'train' in file:

        train_annotation = mmcv.load(file)

        if img_id == None:
            img_id = 0
            for img in train_annotation['images']:
                img_id = max(img_id,img['id'])

        if anno_id == None:
            anno_id = 0
            for anno in train_annotation['annotations']:
                anno_id = max(anno_id,anno['id'])

        dirname = os.path.dirname(os.path.dirname(file))
        imgfolder_train = os.path.join(dirname, 'train')
        imgfolder_valid = os.path.join(dirname, "valid")
        paths_ = []
        for i, (img, _ , paths) in enumerate(dataloader):
            # modify annotaion file
            
            # batch
            for ind in range(len(paths)):
                img_id += 1
                train_annotation['images'].append(dict(id=img_id, width=img[ind].width,
                                                       height=img[ind].height, file_name=paths[ind]))

                if trick == 'threshold':
                    # （M,)
                    scores = output_all[i][2]
                    if eval(args.debug):
                        index = scores > 0
                    else:
                        index = scores > 0.6
                    scores_.append(scores)
                    for sco in scores:
                        score_all_.append(sco)
                    bboxes = output_all[i][0][index]
                    labels = output_all[i][1][index]
                    img___ = mmcv.imread(np.array(img[0]))
                    
                    a = mmcv.imshow_det_bboxes(img___,bboxes,labels,show=False,bbox_color='blue',thickness=3,
                      font_scale=2)
                    mmcv.imwrite(a,'./result_image/cycle'+str(cycle)+'/image'+str(i)+'cycle'+str(cycle)+'.png')
                    for j in range(len(labels)):  # len(output_all[i][0]) = M  <-> num of bboxes

                        anno_id += 1
                        train_annotation['annotations'].append(dict(id=anno_id, image_id=img_id,
                                                                    category_id=labels[j], area=(bboxes[j,2]-bboxes[j,0])*(bboxes[j,3]-bboxes[j,1]),
                                                                    bbox=xy2wh(bboxes[j]), iscrowd=0))
                        label_all_.append(labels[j])
                    # pdb.set_trace()
                paths_.append(paths[ind])
                # move imgs from valid_folder to train_folder
                shutil.move(os.path.join(imgfolder_valid,paths[ind]),os.path.join(imgfolder_train,paths[ind]))
        scorce_csv = pd.DataFrame({'score':scores_})
        scorce_csv.to_csv('/home/czz/AL/CALD/result/scores_output_cycle'+str(cycle)+'.csv')
        index_score = np.array(score_all_)>0.6
        score_all_0_6 = np.array(score_all_)[index_score]
        percent = len(list(score_all_0_6))/len(score_all_)
        print('cycle:percent(>0.6) = '+str(percent*100)+'%')
        plt.figure()
        plt.hist(np.array(score_all_0_6), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.savefig('./result_image/cycle_0_6_'+str(cycle)+'.png')
        plt.figure()
        plt.hist(np.array(score_all_), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.savefig('./result_image/cycle_'+str(cycle)+'.png')
        paths = list(map(lambda x: x + '\n',paths_))
        with open(os.path.dirname(file) + '/labeled_imgid.txt','w+') as f:
           f.write("".join(paths))
        mmcv.dump(train_annotation,file)
        
        for i in range(1,7):
            print('new class '+str(i)+': '+str(label_all_.count(i)))
        
    return percent

def updateoracle(dataloader, oracle_anno: str = "/data01/gpl/ALDataset/BITVehicle_Dataset/annotations/oracle_imgid.txt",
                 valid_pool = "/data01/gpl/ALDataset/BITVehicle_Dataset/valid/",
                 oracle_pool = "/data01/gpl/ALDataset/BITVehicle_Dataset/oracle/"):
    with open(oracle_anno, "w+") as fp:
        for _, _, paths in dataloader:
            for path in paths:
                fp.write(path)
                fp.write('\n')
                # move img file from valid_pool to oracle_pool
                shutil.move(valid_pool + path,
                            oracle_pool + path)

def main(args):
    torch.cuda.set_device(args.gpu_id)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    # Preparing dataset_labeled
    if 'voc2007' in args.dataset:
        dataset_labeled, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
        dataset_unlabeled, _ = get_dataset(args.dataset, "valid", None, args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "test", get_transform(train=False), args.data_path)
    else:
        # dataset_labeled: img, target, path
        # target = dict(image_id=image_id, annotations=target)
        dataset_labeled, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
        # dataset_labeled, num_classes = get_dataset(args.dataset_labeled, "valid", get_transform(train=True), args.data_path)

        dataset_unlabeled, _ = get_dataset(args.dataset, "valid", None, args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "test", get_transform(train=False), args.data_path)
        # dataset_unlabeled = dataset_test
        # dataset_test, _ = get_dataset(args.dataset_labeled, "testtmp", get_transform(train=False), args.data_path)
    # import pdb; pdb.set_trace()
    print("Creating data loaders")
    # TODO: 初始化训练集以及主动学习人工标注的上?
    num_images_labeled = len(dataset_labeled)
    num_images_unlabeled = len(dataset_unlabeled)
    # pdb.set_trace()
    init_num_ = 500
    try:
        init_num_ = args.initlabeled
    except:
        print('w**********w')
    if 'voc' in args.dataset:
        init_num = init_num_
        budget_num = 500
        if 'retina' in args.model:
            init_num = 1000
            budget_num = 500
    else:
        init_num = init_num_
        budget_num = int(args.budget_num)
        oracle_num = int(args.oracle_num)
        # init_num = 50
        # budget_num = 100
    # indices改为两个数据集单独做索引
    indices_labeled = list(range(num_images_labeled))
    # 保证indeces的唯一�?
    indices_unlabeled = list(range(num_images_unlabeled))
    # 首先只取�?00个做索引
    # indices_unlabeled = list(range(100))
    # 此处的原始indices写法
    # indices = list(range(num_images_labeled))

    # 可以采用shuffle
    random.shuffle(indices_labeled)
    # 此处随机打乱
    random.shuffle(indices_unlabeled)
    # TODO: labeled pool
    labeled_set = indices_labeled[:init_num]
    oracle_set = list()
    # TODO: unlabeled pool
    # 此处�?00个交给unlabeled_set
    if eval(args.debug):
        unlabeled_set = indices_unlabeled[:155]
    else:
        unlabeled_set = indices_unlabeled
    # SubsetRandomSampler: Returns a random permutation of integers from 0 to n - 1.
    # 也就是说这样这里还是会全部遍历的，只是遍历的顺序是随机的
    train_sampler = SubsetRandomSampler(labeled_set)
    # len(train_sampler) 500

    # SequentialSampler: Samples elements sequentially, always in the same order.
    data_loader_test = DataLoader(dataset_test, batch_size=1, sampler=SequentialSampler(dataset_test),
                                  num_workers=args.workers, collate_fn=utils.collate_fn)

    augs = []
    if 'F' in args.augs:
        augs.append('flip')
    if 'C' in args.augs:
        augs.append('cut_out')
    if 'D' in args.augs:
        augs.append('smaller_resize')
    if 'R' in args.augs:
        augs.append('rotation')
    if 'G' in args.augs:
        augs.append('ga')
    if 'S' in args.augs:
        augs.append('sp')

    print("Creating model")
    # 此处的num_classes应该考虑背景类，因为目标检测的框难免框住背景类
    # num_classes (int): number of output classes (including background)
    task_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes+1, min_size=800, max_size=1333)
    task_model.to(device)
    # TODO: 循环开�?
    percent_all=[]
    for cycle in range(args.cycles):
        # aspect_ratio_group_factor = 3
        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset_labeled, k=args.aspect_ratio_group_factor)
            # len(group_ids) 6895
            # It enforces that the batch only contain elements from the same group.
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
            # 500 / 4=125
            # len(train_batch_sampler) 125
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(dataset_labeled, batch_sampler=train_batch_sampler, num_workers=args.workers,
                                                  collate_fn=utils.collate_fn)
#####################################################################################in the last part
        params = [p for p in task_model.parameters() if p.requires_grad]
        task_optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        task_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(task_optimizer, milestones=args.lr_steps,
                                                                 gamma=args.lr_gamma)
        # Start active learning cycles training
        # if args.test_only:
        #     if 'coco' in args.dataset:
        #         coco_evaluate(task_model, data_loader_test)
        #     elif 'voc' in args.dataset:
        #         voc_evaluate(task_model, data_loader_test, args.dataset, False, path=args.results_path)
        #     return
        print("Start training")
        start_time = time.time()
        # TODO: 开始训�?
        if cycle==0:
            total = args.total_epochs
        else:
            total = args.each_epochs
        for epoch in range(args.start_epoch, total):
            train_one_epoch(task_model, task_optimizer, data_loader, device, cycle, epoch, args.print_freq)
            task_lr_scheduler.step()
            # evaluate after pre-set epoch
            # 在调试阶段，此处删除test部分
            if not eval(args.debug):
                if (epoch + 1) == total:
                    if 'coco' in args.dataset:
                        coco_evaluate(task_model, data_loader_test)
                    elif 'voc' in args.dataset:
                        voc_evaluate(task_model, data_loader_test, args.dataset_labeled, False, path=args.results_path)
        if not args.skip and cycle == 0:
            if 'faster' in args.model:
                utils.save_on_master({
                    'model': task_model.state_dict(), 'args': args},
                    os.path.join(args.output_dir, '{}_frcnn_1st.pth'.format(args.dataset)))
            elif 'retina' in args.model:
                utils.save_on_master({
                    'model': task_model.state_dict(), 'args': args},
                    os.path.join(args.output_dir, '{}_retinanet_1st.pth'.format(args.dataset)))
        # 啥也没干呢，又重排一�?
        random.shuffle(unlabeled_set)
        print("Getting stability")
        # 这里是真正开始的地方
        if not args.no_mutual:
            # unlabeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(subset),
            #                               num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)

            # SubsetSequentialSampler:Samples elements sequentially from a given list of indices, without replacement
            unlabeled_loader = DataLoader(dataset_unlabeled, batch_size=1, sampler=SubsetSequentialSampler(unlabeled_set),
                                          num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            # pdb.set_trace()
            uncertainty, _cls_corrs, outputs_all = get_uncertainty(task_model, unlabeled_loader, augs, num_classes, args)
            # labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
            #                             num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            # 此部分应该对uncertainty正向排序，因为这些图特别不准，因此应该放到oracle_pool里面
            arg_oracle = np.argsort(np.array(uncertainty))
            tobe_oracle_set_past = arg_oracle[:int(oracle_num)]
            # array([62,  3, 75, 26, 19, 57, 11, 41,  6, 76])
            tobe_oracle_set = list(torch.tensor(unlabeled_set)[tobe_oracle_set_past].numpy())
            # tensor([66, 58, 14,  1, 48,  2, 31, 73,  0, 70])
            oracle_set += tobe_oracle_set
            # pdb.set_trace()
            oracle_loader = DataLoader(dataset_unlabeled, batch_size=1, sampler=SubsetSequentialSampler(tobe_oracle_set),
                                          num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            if not eval(args.debug):
                try:
                    os.mkdir('./result_image_ora/cycle'+str(cycle))
                except:
                    print('cycle'+str(cycle)+' already exist!')
                score_all_ = []
                scores_ = []
                outputs_all_ora = [outputs_all[i] for i in tobe_oracle_set_past]
                for i, (img, _ , paths) in enumerate(oracle_loader):
                    
                    scores = outputs_all_ora[i][2]
                    scores_.append(scores)
                    for sco in scores:
                            score_all_.append(sco)
                    bboxes = outputs_all_ora[i][0]
                    labels = outputs_all_ora[i][1]
                    img___ = mmcv.imread(np.array(img[0]))
                    
                    a = mmcv.imshow_det_bboxes(img___,bboxes,labels,show=False,bbox_color='blue',thickness=3,
                        font_scale=2)
                    mmcv.imwrite(a,'./result_image_ora/cycle'+str(cycle)+'/image_ora'+str(i)+'cycle'+str(cycle)+'.png')
                scorce_csv = pd.DataFrame({'score':scores_})
                scorce_csv.to_csv('/home/czz/AL/CALD/result/scores_output_ora_cycle'+str(cycle)+'.csv')
                plt.figure()
                plt.hist(np.array(score_all_), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
                plt.savefig('./result_image_ora/cycle_'+str(cycle)+'.png')
            
            updateoracle(oracle_loader)
            # len(oracle_loader) 100
            # with open("/data01/zyh/ALDataset/BITVehicle_Dataset/annotations/oracle_imgid.txt","w+") as fp:
            #     for _,_,paths in oracle_loader:
            #         for path in paths:
            #             fp.write(path)
            #             fp.write('\n')
            #             # move img file from valid_pool to oracle_pool
            #             shutil.move("/data01/zyh/ALDataset/BITVehicle_Dataset/valid/"+path,
            #                         "/data01/zyh/ALDataset/BITVehicle_Dataset/oracle/"+path)


            # TODO:此处应该对uncertainty反向排序，用作labeled_pool的一部分
            arg = np.argsort(-np.array(uncertainty))
            cls_corrs_set = arg[:int(args.mr * budget_num)] # mutual range
            # array([58, 66, 47, 15, 64, 55, 32, 59, 56, 13, 31, 33])
            cls_corrs = [_cls_corrs[i] for i in cls_corrs_set]
            # labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
            #                             num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            labeled_loader = DataLoader(dataset_labeled, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
                                        num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            tobe_labeled_set_past = cls_kldiv(labeled_loader, cls_corrs, budget_num, cycle)
            tobe_labeled_set = list(torch.tensor(unlabeled_set)[cls_corrs_set][tobe_labeled_set_past].numpy())

            # outputs_all = list(torch.tensor(outputs_all)[arg].numpy())
            # pdb.set_trace()
            outputs_all = [outputs_all[i] for i in cls_corrs_set[tobe_labeled_set_past]]
            # pdb.set_trace()
            # len(tobe_labeled_set) 100
            tobe_labeled_set_loader = DataLoader(dataset_unlabeled, batch_size=1, sampler=SubsetSequentialSampler(tobe_labeled_set),
                                          num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            # TODO:此处应该往labeled_pool里面加上tobe_labeled_set部分
            # idx_img = []
            # idx_annotation = []
            # pdb.set_trace()
            
            percent = save2file(tobe_labeled_set_loader,output_all=outputs_all,cycle = cycle,args=args)
            percent_all.append(percent)
            # fp_new = open("/data01/zyh/ALDataset/BITVehicle_Dataset/annotations/train_annotation_new.json","w")
            # with open("/data01/zyh/ALDataset/BITVehicle_Dataset/annotations/train_annotation_new.json","r") as f:
            #     data = json.load(f)
            #     for index in data["images"]:
            #         idx_img.append(index["id"])
            #     max_id_img = max(idx_img) + 1
            #     for index in data["annotations"]:
            #         idx_annotation.append(index["id"])
            #     max_id_anno = max(idx_annotation) + 1
            #     for i ,(imgs, _, paths) in enumerate(tobe_labeled_set_loader):
            #         # TODO: file_name
            #         data["images"].append({'id':max_id_img,'width':imgs[0].width,'height':imgs[0].height,'file_name':paths[0]})
            #         # pdb.set_trace()
            #         # 此处只加入了第一个框和对应的类别，原则上这里应该改用score做阈�?
            #         for index in range(len(outputs_all[i][0])):
            #             # box label score
            #             data["annotations"].append({'id':max_id_anno,'image_id':max_id_img,'category_id':outputs_all[i][1][index],'bbox':outputs_all[i][0][index]})
            #             max_id_anno += 1
            #             break
            #         max_id_img += 1
            #     json.dump(data, fp_new, cls=NpEncoder)
            #     # json.dumps(data, cls=NpEncoder)
            # fp_new.close()
            # pdb.set_trace()
            # Update the labeled dataset_labeled and the unlabeled dataset_labeled, respectively

            unlabeled_set = list(set(unlabeled_set) - set(tobe_oracle_set) - set(tobe_labeled_set))
            print('unlabeled_set : '+str(len(unlabeled_set)))
            # TODO: 此处unlabeled_set不应该改成连续的，应该保持离散状�?
            # unlabeled_set = list(range(len(unlabeled_set)))
            # TODO: 此处labeled_set可以改成连续状态，因为在save2file的时候是按照+1的模型往里写�?
            labeled_set += list(range(len(labeled_set),len(labeled_set)+len(tobe_labeled_set)))
            print('labeled_set : '+str(len(labeled_set)))
            # TODO:仿照COCO的json格式完整写下来，因为此处unlabeled_loader是完整的
            # with open("/data01/zyh/ALDataset/BITVehicle_Dataset/annotations/valid_annotation_new.json","w") as fp:
            #     dataset_labeled = {'images': [], 'categories': [], 'annotations': []}
            #     for i ,(imgs,_,paths) in enumerate(unlabeled_loader):
            #         dataset_labeled['images'].append({'id':i,'width':imgs[0].width,'height':imgs[0].height,'path':paths[0]})
            #     json.dump(dataset_labeled,fp, cls=NpEncoder)
            # pdb.set_trace()
        else:

            unlabeled_loader = DataLoader(dataset_unlabeled, batch_size=1, sampler=SubsetSequentialSampler(unlabeled_set),
                                          num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
            uncertainty, _ = get_uncertainty(task_model, unlabeled_loader, augs, num_classes)
            arg = np.argsort(np.array(uncertainty))
            # Update the labeled dataset_labeled and the unlabeled dataset_labeled, respectively
            labeled_set += list(torch.tensor(unlabeled_set)[arg][:budget_num].numpy())
            labeled_set = list(set(labeled_set))
            unlabeled_set = list(set(indices) - set(labeled_set))
        
        dataset_labeled, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
        dataset_unlabeled, _ = get_dataset(args.dataset, "valid", None, args.data_path)
        dataset_test, _ = get_dataset(args.dataset, "test", get_transform(train=False), args.data_path)
        # Create a new dataloader for the updated labeled dataset_labeled
        train_sampler = SubsetRandomSampler(labeled_set)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        # mmcv.imshow_det_bboxes()
        #a = mmcv.imshow_det_bboxes(img,boxes,np.array(['car']))
        #mmcv.imwrite(a,'D:\\Dian\\test.png')
    plt.figure()
    plt.plot(list(range(args.cycles)), percent_all, 'rs-', markersize=10)
    plt.savefig('./test_all_model.png')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--budget-num', default=60, help='num of budget')
    parser.add_argument('--oracle-num', default=50, help='num of oracle')
    parser.add_argument('--initlabeled', default=980, help='num of init labeled')
    parser.add_argument('-d', '--debug', default='False', help='debug quickly')
    parser.add_argument('-p', '--data-path', default='/data01/gpl/ALDataset/BITVehicle_Dataset/', help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-a', '--augs', default='FCDR', help='augmentations')
    # parser.add_argument('-cp', '--first-checkpoint-path', default='/data/yuweiping/coco/',
    #                     help='path to save checkpoint of first cycle')
    parser.add_argument('--task_epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-e', '--total_epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--each_epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--cycles', default=10, type=int, metavar='N',
                        help='number of cycles epochs to run')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.0025, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--ll-weight', default=0.5, type=float,
                        help='ll loss weight')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 19], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=1000, type=int, help='print frequency')
    # TODO: output_dir / results
    parser.add_argument('--output-dir', default='out1', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('-rp', '--results-path', default='results',
                        help='path to save detection results (only for voc)')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('-i', "--init", dest="init", help="if use init sample", action="store_true")
    parser.add_argument('-u', "--uniform", dest="uniform", help="if use init sample", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument('-s', "--skip", dest="skip", help="Skip first cycle and use pretrained model to save time",
                        action="store_true")
    parser.add_argument('-m', '--no-mutual', help="without mutual information",
                        action="store_true")
    parser.add_argument('-mr', default=1.2, type=float, help='mutual range')
    parser.add_argument('-bp', default=1.3, type=float, help='base point')
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true")
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--gpu-id',default = '0',type = int,help = 'gpu-id')
    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)






            # pdb.set_trace()
        # len(data_loader) 125
        # 原始创建模型写法
        # print("Creating model")
        # if 'voc' in args.dataset:
        #     if 'faster' in args.model:
        #         task_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes, min_size=600, max_size=1000)
        #     elif 'retina' in args.model:
        #         task_model = retinanet_resnet50_fpn_cal(num_classes=num_classes, min_size=600, max_size=1000)
        # else:
        #     if 'faster' in args.model:
        #         # 此处的num_classes应该考虑背景类，因为目标检测的框难免框住背景类
        #         task_model = fasterrcnn_resnet50_fpn_feature(num_classes=num_classes+1, min_size=800, max_size=1333)
        #     elif 'retina' in args.model:
        #         task_model = retinanet_resnet50_fpn_cal(num_classes=num_classes, min_size=800, max_size=1333)
        # task_model.to(device)
        # if cycle == 0 and args.skip:
        #     if 'faster' in args.model:
        #         checkpoint = torch.load(os.path.join(args.output_dir,
        #                                              '{}_frcnn_1st.pth'.format(args.dataset)), map_location='cpu')
        #     elif 'retina' in args.model:
        #         checkpoint = torch.load(os.path.join(args.output_dir,
        #                                              '{}_retinanet_1st.pth'.format(args.dataset)), map_location='cpu')
        #     task_model.load_state_dict(checkpoint['model'])
        #     if args.test_only:
        #         if 'coco' in args.dataset:
        #             coco_evaluate(task_model, data_loader_test)
        #         elif 'voc' in args.dataset:
        #             voc_evaluate(task_model, data_loader_test, args.dataset, False, path=args.results_path)
        #         return
        #     # 开始检�?
        #     print("Getting stability")
        #     random.shuffle(unlabeled_set)
        #     if not args.no_mutual:
        #         unlabeled_loader = DataLoader(dataset_unlabeled, batch_size=1, sampler=SubsetSequentialSampler(unlabeled_set),
        #                                       num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        #         # TODO: Metric
        #         uncertainty, _cls_corrs,outputs_all = get_uncertainty(task_model, unlabeled_loader, augs, num_classes)

        #         # arg_oracle = np.argsort(np.array(uncertainty))
        #         # tobe_oracle_set = arg_oracle[:int(oracle_num)]
        #         # tobe_oracle_set = list(torch.tensor(unlabeled_set)[tobe_oracle_set].numpy())
        #         # oracle_set += tobe_oracle_set
        #         # oracle_loader = DataLoader(dataset_unlabeled, batch_size=1,
        #         #                            sampler=SubsetSequentialSampler(tobe_oracle_set),
        #         #                            num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        #         # updateoracle(oracle_loader)


        #         # 首先按照uncertainty从小到大排序
        #         arg = np.argsort(np.array(uncertainty))
        #         # 选择一批略大于budget_num的数据，得到对应的类别分�?
        #         cls_corrs_set = arg[:int(args.mr * budget_num)]
        #         cls_corrs = [_cls_corrs[i] for i in cls_corrs_set]

        #         labeled_loader = DataLoader(dataset_labeled, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
        #                                     num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        #         tobe_labeled_set = cls_kldiv(labeled_loader, cls_corrs, budget_num, cycle)
        #         # Update the labeled dataset_labeled and the unlabeled dataset_labeled, respectively
        #         tobe_labeled_set = list(torch.tensor(unlabeled_set)[arg][tobe_labeled_set].numpy())
        #         labeled_set += list(range(len(labeled_set),len(labeled_set)+len(tobe_labeled_set)))
        #         unlabeled_set = list(set(unlabeled_set) - set(tobe_labeled_set))

        #         # arg = np.argsort(-np.array(uncertainty))
        #         # cls_corrs_set = arg[:int(args.mr * budget_num)]  # mutual range
        #         # cls_corrs = [_cls_corrs[i] for i in cls_corrs_set]
        #         # labeled_loader = DataLoader(dataset_aug, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
        #         #                             num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        #         # labeled_loader = DataLoader(dataset_labeled, batch_size=1, sampler=SubsetSequentialSampler(labeled_set),
        #         #                             num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        #         # tobe_labeled_set = cls_kldiv(labeled_loader, cls_corrs, budget_num, cycle)
        #         # tobe_labeled_set = list(torch.tensor(unlabeled_set)[arg][tobe_labeled_set].numpy())
        #         # outputs_all = [outputs_all[i] for i in arg]
        #         # tobe_labeled_set_loader = DataLoader(dataset_unlabeled, batch_size=1,
        #         #                                      sampler=SubsetSequentialSampler(tobe_labeled_set),
        #         #                                      num_workers=args.workers, pin_memory=True,
        #         #                                      collate_fn=utils.collate_fn)
        #         # save2file(tobe_labeled_set_loader, output_all=outputs_all)
        #         # unlabeled_set = list(set(unlabeled_set) - set(tobe_oracle_set) - set(tobe_labeled_set))
        #         # unlabeled_set = list(range(len(unlabeled_set)))
        #         # unlabeled_loader = DataLoader(dataset_unlabeled, batch_size=1,
        #         #                               sampler=SubsetSequentialSampler(unlabeled_set),
        #         #                               num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        #         # labeled_set += list(range(len(labeled_set), len(labeled_set) + len(tobe_labeled_set)))
        #         # print("first cycle finished!")

        #     else:
        #         unlabeled_loader = DataLoader(dataset_unlabeled, batch_size=1, sampler=SubsetSequentialSampler(unlabeled_set),
        #                                       num_workers=args.workers, pin_memory=True, collate_fn=utils.collate_fn)
        #         uncertainty, _,_ = get_uncertainty(task_model, unlabeled_loader, augs, num_classes)
        #         arg = np.argsort(np.array(uncertainty))
        #         # Update the labeled dataset_labeled and the unlabeled dataset_labeled, respectively
        #         labeled_set += list(torch.tensor(unlabeled_set)[arg][:budget_num].numpy())
        #         labeled_set = list(set(labeled_set))
        #         unlabeled_set = list(set(unlabeled_set) - set(tobe_labeled_set))

        #     # Create a new dataloader for the updated labeled dataset_labeled
        #     train_sampler = SubsetRandomSampler(labeled_set)
        #     continue
