import os
import pdb
import torch
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders.dataset_2d import *
from configs.config_mambaunet import get_config
from networks.mambaunet.mambaunet import MambaUnet
from networks.mambaunet.mambaunet_dia import MambaUnet_dia
from utils.val_2D import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz/', help='data path')
parser.add_argument('--exp', type=str, default='dcmamba', help='experiment name')
parser.add_argument('--network', type=str, default='mambaunet', help='model name')
parser.add_argument('--dataset', type=str,  default='synapse', help='dataset to use')
parser.add_argument('--num_classes', type=int,  default=9, help='output channel of network')
parser.add_argument('--iteration', type=int,  default=20000, help='GPU to use')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--load_ckpt_path', type=str, default='./pre_trained_weights/vmamba_small_e238_ema.pth', help='Path of ckpt')

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

parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
root = "../"

snapshot_path = root + "model_" + args.dataset + "/" + args.exp + "/"
test_save_path = root + "model_" + args.dataset + "/prediction/" + args.exp + "_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = args.num_classes
if args.dataset == 'synapse':
    args.patch_size = [224, 224]
    with open(args.root_path + '/../test_vol.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + '/' + item.replace('\n', '') + ".npy.h5" for item in image_list]
elif "isic" in args.dataset:
    test_transformer = transforms.Compose([myNormalize(args.dataset, train=False), myToTensor(), myResize(args.patch_size[0], args.patch_size[1])])
    val_dataset = NPY_datasets(args.root_path, train=False, transform=test_transformer)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
elif args.dataset == 'kvasir':
    num_chns = 3
    test_transformer = transforms.Compose([myNormalize(args.dataset, train=False), myToTensor(), myResize(args.patch_size[0], args.patch_size[1])])
    val_dataset = Kvasir(args.root_path, train=False, transform=test_transformer)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

if __name__ == '__main__':
    if args.network == 'mambaunet':
        config = get_config(args)
        model = MambaUnet(config, img_size=args.patch_size, num_classes=num_classes).cuda()
    elif args.network == 'mambaunet_dia':
        config = get_config(args)
        model = MambaUnet_dia(config, img_size=args.patch_size, num_classes=num_classes).cuda()

    try:
        model.load_state_dict(torch.load(os.path.join(snapshot_path, 'iter_' + str(args.iteration) + '.pth')))
    except:
        model.load_state_dict(torch.load(os.path.join(snapshot_path, 'model2_iter_' + str(args.iteration) + '.pth')))
    model.eval()

    if args.dataset == 'synapse':
        metric, std = test_all_case(model, args.dataset, image_list, num_classes=num_classes, patch_size=args.patch_size, save_result=True, test_save_path=test_save_path)
        avg_dice = np.mean(metric[0])
        with open(root + "model_" + args.dataset + "/prediction.txt", "a") as f:
            f.write(args.exp + " - " + str(args.iteration) + ": " + ", ".join(str(i) for i in metric) + "\n")
            f.write(args.exp + " - " + str(args.iteration) + " avg dice: " + str(avg_dice) + "\n")
            f.write(args.exp + " - " + str(args.iteration) + ": " + ", ".join(str(i) for i in std) + "\n")
    else:
        metric = test_all_case_isic(model, args.dataset, val_loader, save_result=True, test_save_path=test_save_path)
        with open(root + "model_" + args.dataset + "/prediction.txt", "a") as f:
            f.write(args.exp + " - " + str(args.iteration) + ": " + ", ".join(str(i) for i in metric) + "\n")
