import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
import transforms as T
import utils
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tqdm
from torch.autograd import Variable
from skp_rpn_dataloader import Kp_Range_Dataset
from skp_3d_detnet import SKP3DDetNet


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for skp, positions, sizes, anchor_vecs in metric_logger.log_every(data_loader, print_freq, header):
        skp = skp.to(device)
        positions = positions.to(device)
        sizes = sizes.to(device)
        anchor_vecs = anchor_vecs.to(device)
        loss_dict = model(skp, positions, sizes, anchor_vecs)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

def main(args):
    
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset = Kp_Range_Dataset('/media/beta/新加卷/Labelled_Dataset/kitti3d/skp/kp2/', '/media/beta/新加卷/Labelled_Dataset/kitti3d/skp/label2/', args.min_x, args.max_x, args.min_z, args.max_z, args.step)
    # dataset_test = Kp_Range_Dataset('/media/beta/新加卷/SG_kpval/', '/home/beta/SG2020/kitti_3d/train_set/pose/', '/home/beta/SG2020/kitti_3d/test_set/pose/', args.min_x, args.max_x , args.min_z, args.max_z)

    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
       #  test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
       #  test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, drop_last=True)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1,
    #     sampler=test_sampler, num_workers=args.workers)

    model = SKP3DDetNet(args.min_x, args.max_x, args.min_z, args.max_z, args.step, args.fg_dis)
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # if args.test_only:
    #     confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
    #     print(confmat)
    #     return

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
    ]
    
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
        # print(confmat)
        if epoch % 5 == 0:
            if args.output_dir:
                utils.mkdir(args.output_dir)
            utils.save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args
                },
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        # evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('--dataset', default='voc', help='dataset')
    parser.add_argument('--model', default='fcn_resnet101', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=130, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--min-x', default=-25, type=int, help='print frequency')
    parser.add_argument('--max-x', default=25, type=int, help='print frequency')
    parser.add_argument('--min-z', default=0, type=int, help='print frequency')
    parser.add_argument('--max-z', default=35, type=int, help='print frequency')
    parser.add_argument('--step', default=5, type=int, help='print frequency')
    parser.add_argument('--fg-dis', default=8, type=int, help='print frequency')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)