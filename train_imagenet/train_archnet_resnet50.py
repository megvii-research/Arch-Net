import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import argparse
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import logging
import datetime
import shlex
import subprocess
import time
import random
import sys
sys.path.append("..")
from models.students.resnet50 import archnet_resnet50 as student_model_function
from models.teachers.resnet import resnet50 as teacher_model_function
from datasets.imagenet_dataset import CustomImagenetDataset as imagenet_dataset
from evaluate.imagenet_validataion import imagenet_validation


torch.distributed.init_process_group(backend="nccl", world_size=8)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

parser = argparse.ArgumentParser(description="PyTorch BrigeNet-ResNet50 ImageNet Distillation")

parser.add_argument(
    "--weight-bit",
    default=2,
    type=int,
    help="weight bit",
)

parser.add_argument(
    "--feature-bit",
    default=4,
    type=int,
    help="feature bit",
)

parser.add_argument(
    "-j",
    "--workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)

parser.add_argument(
    "--validate-workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers during validation (default: 16)",
)

parser.add_argument(
    "--total-images-per-period",
    default=8192,
    type=int,
    help="total number of training images",
)

parser.add_argument(
    "--total-images-all-periods",
    default=30000,
    type=int,
    help="total number of training images",
)

parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="BS",
    help="mini-batch size (default: 256)",
)

parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
)

parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    metavar="M",
    help="momentum"
)

parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-7,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)

parser.add_argument(
    "--print-frequency",
    default=1,
    type=int,
    metavar="N",
    help="print loss log every N mini-batches (default: 8)",
)

parser.add_argument('--local_rank', default=0, type=int, help='local rank')

parser.add_argument('--num_gpus', default=8, type=int, help='number of gpus')

parser.add_argument(
    "--first-period-epochs", default=60, type=int, metavar="N", help="number of total epochs of the first period"
)

parser.add_argument(
    "--last-period-epochs", default=5110, type=int, metavar="N", help="number of total epochs of the last period"
)

parser.add_argument(
    "--middle-period-epochs", default=10, type=int, metavar="N", help="number of the epochs of each middle period"
)

parser.add_argument(
    "--number-of-middle-periods", default=64, type=int, metavar="N", help="number of the middle periods, except for the first and the last period"
)

parser.add_argument(
    "--output-path", type=str, default="../output_results/archnet_resnet50", help="a folder save training log"
)

parser.add_argument(
    "--teacher-model-path", type=str, default="../models/teachers/pretrained_models/resnet50-19c8e357.pth", help="path of the teacher model"
)

parser.add_argument(
    "--train-data-path", type=str, default="../data/imagenet/train/", help="path of the training data"
)

parser.add_argument(
    "--val-data-path", type=str, default="../data/imagenet/val/", help="path of the validate data"
)

parser.add_argument(
    "--evaluate", action="store_true", help="validate"
)

parser.add_argument(
    "--restart-period", default=0, type=int, help="the period that you want to restart",
)

parser.add_argument(
    "--sync-bn", action="store_true", help="use sync bn or not",
)


def get_teacher_model(teacher_model_path):
    teacher_model_state_dict = torch.load(teacher_model_path, map_location='cpu')
    teacher_model = teacher_model_function()
    teacher_model.load_state_dict(teacher_model_state_dict)

    return teacher_model


def get_student_model(teacher_model_path, idx, saved_model_path, training_periods, args):
    student_model = student_model_function(args.weight_bit, args.feature_bit, squeeze_factor=24, distillation_idx=idx, is_train=True)
    student_model_state_dict = student_model.state_dict()

    if idx == 0:
        teacher_model_state_dict = torch.load(teacher_model_path, map_location='cpu')
        for k in student_model_state_dict.keys():
            if k == 'conv1.conv_Q.weight' or ('layer' not in k and 'bn1' in k):
                continue
            elif k in teacher_model_state_dict:
                student_model_state_dict[k] = teacher_model_state_dict[k]
            # conv weight and bias
            elif '.conv_Q' in k and 'concat' not in k and 'fc' not in k:
                k_split = k.split('.conv_Q')
                new_k = ''.join(k_split)
                if new_k in teacher_model_state_dict:
                    if 'downsample' in k and 'layer4' not in k:  # downsample in last stage is different
                        student_v = student_model_state_dict[k]
                        student_v[:] = 1e-10
                        student_v[:, :, 1:2, 1:2] = teacher_model_state_dict[new_k]
                        student_model_state_dict[k] = student_v
                    elif 'downsample' not in k:
                        teacher_v = teacher_model_state_dict[new_k]
                        if teacher_v.shape[-1] == 3:  # kernel size = 3
                            student_model_state_dict[k] = teacher_v
                        elif teacher_v.shape[-1] == 1:  # kernel size = 1
                            student_v = student_model_state_dict[k]
                            student_v[:] = 1e-10
                            student_v[:, :, 1:2, 1:2] = teacher_model_state_dict[new_k]
                            student_model_state_dict[k] = student_v
            # elif 'fc.linear_Q.weight' in k:
            #     student_model_state_dict[k] = teacher_model_state_dict['fc.weight']
            # elif 'fc.linear_Q.bias' in k:
            #     student_model_state_dict[k] = teacher_model_state_dict['fc.bias']
    else:
        distillation_model_path = saved_model_path + '/period_{}_epoch_{}.pth'.format(idx-1, training_periods[idx-1])
        student_model_state_dict_multigpus = torch.load(distillation_model_path, map_location='cpu')
        for k, v in student_model_state_dict_multigpus.items():
            student_model_state_dict[k[7:]] = v

    student_model.load_state_dict(student_model_state_dict)

    return student_model


def mixup_data(batch, alpha):
    lam = np.random.beta(alpha, alpha)
    batch_size = batch.shape[0]
    idx = torch.randperm(batch_size)
    mixup_batch = lam * batch + (1 - lam) * batch[idx]

    return mixup_batch


def get_loss(outputs, labels, last_period=False):
    assert len(outputs) == len(labels)

    loss_fn = nn.MSELoss()
    len_outputs = len(outputs)
    if last_period:
        multiplier = torch.logspace(len_outputs, 1, len_outputs, base=0.8)
    else:
        multiplier = torch.logspace(len_outputs-1, 0, len_outputs, base=0.8)

    loss = 0
    for i in range(len_outputs):
        loss += multiplier[i] * loss_fn(outputs[i], labels[i])  # default is F norm

    return loss


def cosine_loss(outputs, labels):
    assert len(outputs) == len(labels)

    len_outputs = len(outputs)
    multiplier = torch.logspace(len_outputs-1, 0, len_outputs, base=0.8)

    loss = 0
    for i in range(len_outputs):
        loss_i = 1.0 - torch.mean(torch.cosine_similarity(outputs[i], labels[i]))
        loss += multiplier[i] * loss_i

    return loss


def validate(model_dir, training_periods, logger, workers, args):
    # build new model
    new_model = student_model_function(args.weight_bit, args.feature_bit, squeeze_factor=24, distillation_idx=None, is_train=False)
    new_model_state_dict = new_model.state_dict()
    last_period = 'period_{}'.format(len(training_periods) - 1)
    for path in os.listdir(model_dir):
        if last_period in path:
            model_name, period, epoch = get_print_info(path)
            model_path = os.path.join(model_dir, path)
            model_state_dict = torch.load(model_path, map_location='cpu')
            for k, v in model_state_dict.items():
                new_model_state_dict[k[7:]] = v
            new_model.load_state_dict(new_model_state_dict)

            # validate
            print('Start evaluating model: ' + model_name)
            prec1, prec5 = imagenet_validation(new_model, args.val_data_path, num_threads=workers, model_name=model_name, batch_size=32)
            info_str = f'Period: {period} Epoch: {epoch} | top-1: {prec1} | top-5: {prec5}'
            logger.info(info_str)


def get_print_info(model_path):
    model_name = model_path.split('.')[0]
    period_and_epoch = model_name.split('_')
    period = period_and_epoch[1]
    epoch = period_and_epoch[-1]

    return model_name, period, epoch


def construct_logger(name, save_dir):
    def git_hash():
        cmd = 'git log -n 1 --pretty="%h"'
        ret = subprocess.check_output(shlex.split(cmd)).strip()
        if isinstance(ret, bytes):
            ret = ret.decode()
        return ret

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    date = str(datetime.datetime.now().strftime('%m%d%H'))
    fh = logging.FileHandler(os.path.join(save_dir, f'log-{date}-{git_hash()}.txt'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def main():
    start_time = datetime.datetime.now()
    args = parser.parse_args()

    training_periods = [args.first_period_epochs] + [args.middle_period_epochs for _ in range(args.number_of_middle_periods)] + [args.last_period_epochs]
    save_dir = args.output_path + '_lr' + str(args.lr).replace('.', '_') + '_bs' + str(args.batch_size) + '_images' + str(args.total_images_per_period)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    logger = construct_logger('resnet50', save_dir)

    if args.evaluate:
        if local_rank == 0:
            print('Start Validating...')
            validate(save_dir, training_periods, logger, args.validate_workers, args)
        return

    # get teacher model
    teacher_model = get_teacher_model(args.teacher_model_path)
    teacher_model = teacher_model.to(device)
    teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    teacher_model.eval()

    # prepare datalaoder
    ds = imagenet_dataset(args.train_data_path, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]), num_images=args.total_images_all_periods)
    sampler = DistributedSampler(ds)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)
    iters_per_epoch = args.total_images_per_period // args.batch_size // args.num_gpus

    batch_time = AverageMeter()
    total_epochs = sum(training_periods[args.restart_period:])
    current_total_epochs = 0
    for idx, training_period in enumerate(training_periods):
        if idx < args.restart_period:
            continue
        # get student model and its middle output
        student_model = get_student_model(args.teacher_model_path, idx, save_dir, training_periods, args)
        torch.distributed.barrier()
        student_model = student_model.to(device)
        if args.sync_bn:
            process_group = torch.distributed.new_group()
            student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model, process_group)
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        student_model.train()

        cudnn.benchmark = True

        # optimizer
        opt = optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # distillation
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

        if local_rank == 0:
            print('Start Training Period {}, {} epochs in total'.format(idx, training_period))
        for epoch in range(training_period):
            current_total_epochs += 1
            running_loss = 0.0
            end = time.time()
            sampler.set_epoch(current_total_epochs)
            for new_idx, sample in enumerate(train_loader):
                if new_idx > args.total_images_per_period // args.batch_size // args.num_gpus - 1:
                    break
                inputs = sample[0]
                if random.random() > 0.9:
                    inputs = mixup_data(inputs, 1.0)
                inputs = inputs.cuda()
                with torch.no_grad():
                    labels = teacher_model(inputs)

                opt.zero_grad()
                student_model_middle_output = student_model([labels, inputs])
                if idx == len(training_periods) - 1:
                    loss = get_loss(student_model_middle_output[:-1], labels[:-1], last_period=True) + cosine_loss([student_model_middle_output[-1]], [labels[-1]])
                else:
                    loss = get_loss(student_model_middle_output, labels[:idx+1])
                loss.backward()
                opt.step()

                batch_time.update(time.time() - end)
                end = time.time()
                eta = batch_time.avg * (iters_per_epoch - 1 - new_idx + iters_per_epoch * (total_epochs - current_total_epochs - 1)) // 1

                # print statistics
                running_loss += loss.item()
                if new_idx % args.print_frequency == args.print_frequency - 1 and local_rank == 0:  # print every 1 mini-batch
                    print(
                        'Period/Epoch/Iteration: [{0}/{1}/{2}] '
                        'Loss: {loss:.10f} '
                        'Eta: {eta}'.format(
                            idx,
                            epoch + 1,
                            new_idx + 1,
                            loss=running_loss / args.print_frequency,
                            eta=str(datetime.timedelta(seconds=eta)),
                        ))
                    running_loss = 0.0
            scheduler.step()

            if idx < len(training_periods) - 1:
                if epoch == training_period - 1 and local_rank == 0:
                    model_path = 'period_{}_epoch_{}.pth'.format(idx, epoch + 1)
                    model_save_path = os.path.join(save_dir, model_path)
                    torch.save(student_model.state_dict(), model_save_path)

                    if idx > 0 and idx < len(training_periods) - 1 and local_rank == 0:
                        delete_model_path = 'period_{}_epoch_{}.pth'.format(idx-1, training_periods[idx-1])
                        delete_model_save_path = os.path.join(save_dir, delete_model_path)
                        if os.path.exists(delete_model_save_path):
                            os.remove(delete_model_save_path)
            elif local_rank == 0:
                if epoch > args.last_period_epochs - 20:
                    model_path = 'period_{}_epoch_{}.pth'.format(idx, epoch + 1)
                    model_save_path = os.path.join(save_dir, model_path)
                    torch.save(student_model.state_dict(), model_save_path)
            torch.distributed.barrier()

    end_time = datetime.datetime.now()
    if local_rank == 0:
        print('The torch random seed of the training procedure is {}'.format(torch.initial_seed()))
        print('It took {} to train'.format(end_time - start_time))
        print('The save path is ' + save_dir)
        print('Finish Training. Start Validating...')
        validate(save_dir, training_periods, logger, args.validate_workers, args)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()

# 'rlaunch --cpu 16 --gpu 8 memory 65536'
# 'python3 -m torch.distributed.launch --nproc_per_node=8 filename -j 16' to run
