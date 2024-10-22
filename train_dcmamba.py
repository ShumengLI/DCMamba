import os
import gc
import pdb
import sys
import random
import shutil
import logging
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from itertools import cycle

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from dataloaders.dataset_2d import *
from dataloaders.StrongAug_2d import get_StrongAug_pixel
from configs.config_mambaunet import get_config
from networks.mambaunet.mambaunet_dir import MambaUnet
from networks.mambaunet.mambaunet_dia_dir import MambaUnet_dia
from networks.utils import projectors
from utils import losses, ramps
from utils.val_2D import save_imgs

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz/', help='data path')
parser.add_argument('--exp', type=str, default='dcmamba', help='experiment name')
parser.add_argument('--network1', type=str, default='mambaunet_dia', help='model1 name')
parser.add_argument('--network2', type=str, default='mambaunet', help='model2 name')
parser.add_argument('--dataset', type=str, default='synapse', help='dataset to use')

parser.add_argument('--labeled_num', type=int, default=4, help='labeled data')
parser.add_argument('--batch_size', type=int, default=24, help='batch size per gpu')
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled batch size per gpu')
parser.add_argument('--num_classes', type=int,  default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--p_per_sample', type=float, default=0.9, help='p_per_sample')

parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--patch_mix_size', type=int, default=4, help='patch size of mix input')
# pretrain
parser.add_argument('--load_ckpt_path', type=str, default='./pre_trained_weights/vmamba_tiny_e292.pth', help='Path of ckpt')
# mambaunet/swinunet config
parser.add_argument('--cfg', type=str, default="./configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument("--opts", default=None, help="Modify config options by adding 'KEY VALUE' pairs. ", nargs='+',)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument('--save_img', type=int, default=500, help='img saving iterations')
parser.add_argument('--save_model', type=int, default=10000, help='model saving iterations')

parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# Network definition
def create_model(network):
    if network == 'mambaunet':
        config = get_config(args)
        model = MambaUnet(config, img_size=args.patch_size, num_classes=args.num_classes).cuda()
        model.load_from(config)
    elif network == 'mambaunet_dia':
        config = get_config(args)
        model = MambaUnet_dia(config, img_size=args.patch_size, num_classes=args.num_classes).cuda()
        model.load_from(config)
    return model

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def patch_wise_mix(batch_a, batch_b, patch_size=4):
    batch_size, channels, height, width = batch_a.shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    batch_a = batch_a.view(batch_size, channels, patch_height, patch_size, patch_width, patch_size)
    batch_b = batch_b.view(batch_size, channels, patch_height, patch_size, patch_width, patch_size)
    mask = torch.rand(batch_size, 1, patch_height, 1, patch_width, 1) < 0.5
    mixed_a = torch.where(mask, batch_b, batch_a)
    mixed_b = torch.where(mask, batch_a, batch_b)
    mixed_a = mixed_a.view(batch_size, channels, height, width)
    mixed_b = mixed_b.view(batch_size, channels, height, width)
    return mixed_a, mixed_b

def DFF(x):
    x_var = x.var(dim=0)
    x_weight = 1 / (1 + torch.exp(-x_var))
    x_sum = torch.sum(x, dim=0, keepdim=True)
    x_fs = x_sum * x_weight
    x = x_fs.squeeze(dim=0)
    return x


def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    p_per_sample = args.p_per_sample

    weak_synapse = transforms.Compose([RandomGenerator(args.patch_size)])
    weak = transforms.Compose([
            myNormalize(args.dataset, train=True),
            myToTensor(),
            myRandomHorizontalFlip(p=0.5),
            myRandomVerticalFlip(p=0.5),
            myRandomRotation(p=0.5, degree=[0, 360]),
            myResize(args.patch_size[0], args.patch_size[1])
        ])
    strong = transforms.Compose([get_StrongAug_pixel(args.patch_size, 3, p_per_sample)])

    labeled_slice = args.labeled_num
    if args.dataset == 'synapse':
        args.patch_size = [224, 224]
        labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
        db_train = Synapse(base_dir=args.root_path, file_name="train", transform=weak_synapse, strong_transform=strong)
    elif "isic" in args.dataset:
        db_train = NPY_datasets(base_dir=args.root_path, transform=weak, strong_transform=strong)
    elif args.dataset == 'kvasir':
        db_train = Kvasir(base_dir=args.root_path, transform=weak, strong_transform=strong)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, len(db_train)))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1 = create_model(args.network1)
    model2 = create_model(args.network2)
    model1.train()
    model2.train()
    projector_1 = projectors(input_nc=1536).cuda()
    projector_2 = projectors(input_nc=1536).cuda()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    contrastive_loss_sup_criter = losses.contrastive_loss_sup()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            names, volume_batch_w, volume_batch_s, label_batch = sampled_batch['name'], sampled_batch['image'], sampled_batch['strong_aug'], sampled_batch['label'].long()

            ### diverse augment
            if args.dataset == "synapse":
                mix_size = random.choice([28, 56, 112, 224])
            else:
                mix_size = random.choice([32, 64, 128, 256])
            mixed_a, mixed_b = patch_wise_mix(volume_batch_w, volume_batch_s, patch_size=mix_size)

            volume_batch_w, volume_batch_s, label_batch = volume_batch_w.cuda(), volume_batch_s.cuda(), label_batch.cuda()
            mixed_a, mixed_b = mixed_a.cuda(), mixed_b.cuda()

            ### diverse scan
            outputs1_w, y_encoder1 = model1(mixed_a)
            outputs_soft1_w = torch.softmax(outputs1_w, dim=1)
            outputs2_s, y_encoder2 = model2(mixed_b)
            outputs_soft2_s = torch.softmax(outputs2_s, dim=1)

            sup_loss1_w = 0.5 * (ce_loss(outputs1_w[:labeled_bs], label_batch[:labeled_bs]) +
                               dice_loss(outputs_soft1_w[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1)))
            sup_loss2_s = 0.5 * (ce_loss(outputs2_s[:labeled_bs], label_batch[:labeled_bs]) +
                                 dice_loss(outputs_soft2_s[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(outputs_soft1_w[labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2_s[labeled_bs:].detach(), dim=1, keepdim=False)

            unsup_loss1 = 0.5 * (ce_loss(outputs1_w[labeled_bs:], pseudo_outputs2) +
                                 dice_loss(outputs_soft1_w[labeled_bs:], pseudo_outputs2.unsqueeze(1)))
            unsup_loss2 = 0.5 * (ce_loss(outputs2_s[labeled_bs:], pseudo_outputs1) +
                                 dice_loss(outputs_soft2_s[labeled_bs:], pseudo_outputs1.unsqueeze(1)))

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            model1_loss = sup_loss1_w + consistency_weight * unsup_loss1
            model2_loss = sup_loss2_s + consistency_weight * unsup_loss2

            ### diverse feature
            feat1 = torch.stack(y_encoder1)
            feat2 = torch.stack(y_encoder2)
            feat1, feat2 = DFF(feat1), DFF(feat2)
            feat1 = projector_1(feat1.permute(0, 3, 1, 2))
            feat2 = projector_2(feat2.permute(0, 3, 1, 2))
            loss_feat = contrastive_loss_sup_criter(feat1, feat2)

            # total loss
            loss = model1_loss + model2_loss + loss_feat

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            if iter_num % args.save_img == 0:
                out = torch.argmax(outputs_soft1_w, dim=1)
                out_s = torch.argmax(outputs_soft2_s, dim=1)
                if args.dataset == "synapse":
                    nib.save(nib.Nifti1Image(mixed_a[0, 0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/img_' + str(iter_num) + '.nii.gz')
                    nib.save(nib.Nifti1Image(label_batch[0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/gt_' + str(iter_num) + '.nii.gz')
                    nib.save(nib.Nifti1Image(out[0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/pred_' + str(iter_num) + '.nii.gz')
                else:
                    msk = label_batch.unsqueeze(1).cpu().detach().numpy()
                    out = out.unsqueeze(1).cpu().detach().numpy()
                    out_s = out_s.unsqueeze(1).cpu().detach().numpy()
                    save_imgs(mixed_a[0], msk[0], out[0], outputs_soft1_w[0].cpu().detach().numpy(), names[0], snapshot_path + '/saveimg/img_' + str(iter_num) + '_aug1_', args.dataset)
                    save_imgs(mixed_b[0], msk[0], out_s[0], outputs_soft2_s[0].cpu().detach().numpy(), names[0], snapshot_path + '/saveimg/img_' + str(iter_num) + '_aug2_', args.dataset)

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer2.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/total_loss', loss, iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss, iter_num)
            writer.add_scalar('loss/loss_feat', loss_feat, iter_num)
            if iter_num % 20 == 0:
                logging.info('iteration %d : model1 loss : %f model2 loss : %f feat loss : %f' % (iter_num, model1_loss.item(), model2_loss.item(), loss_feat.item()))

            # change lr
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            if iter_num % args.save_model == 0:
                save_mode_path = os.path.join(snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))
                save_mode_path = os.path.join(snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    save_mode_path = os.path.join(snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
    torch.save(model1.state_dict(), save_mode_path)
    logging.info("save model1 to {}".format(save_mode_path))
    save_mode_path = os.path.join(snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
    torch.save(model2.state_dict(), save_mode_path)
    logging.info("save model2 to {}".format(save_mode_path))
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model_" + args.dataset + "/" + args.exp + "/"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + 'saveimg')
    shutil.copy('train_dcmamba.py', snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
