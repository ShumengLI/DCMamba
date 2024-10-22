import pdb
import cv2
import h5py
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from medpy import metric
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def save_imgs(img, msk, msk_pred, score, i, save_path, datasets, threshold=0.5):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7,15))

    plt.subplot(5,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(5,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(5,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    plt.subplot(5, 1, 4)
    plt.imshow(score[0])
    plt.axis('off')

    plt.subplot(5, 1, 5)
    plt.imshow(score[1])
    plt.axis('off')

    plt.colorbar()
    plt.savefig(save_path + str(i) +'.png')
    plt.close()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_all_case(net, dataset, image_list, num_classes, patch_size, save_result=True, test_save_path=None):
    total_metric = []
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        if dataset == 'synapse':
            id = image_path.split('.')[-3].split('/')[-1]
        elif dataset == 'acdc':
            id = image_path.split('.')[-2].split('/')[-1]

        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred

        if np.sum(prediction)==0:
            single_metric = np.zeros((4, num_classes-1))
        else:
            single_metric = np.zeros((2, num_classes-1))          # num_classes=8
            for c in range(1, num_classes):
                single_metric[:, c-1] = calculate_metric_percase(prediction == c, label == c)
        print(id, single_metric[0])
        total_metric.append(np.expand_dims(single_metric, axis=0))

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")

    total_array = np.concatenate(total_metric, axis=0)
    avg_metric, std = np.mean(total_array, axis=0), np.std(total_array, axis=0)
    print('average metric is {}'.format(avg_metric))
    print('average std is {}'.format(std))

    return avg_metric, std


def test_all_case_isic(net, dataset, test_loader, threshold=0.5, save_result=True, test_save_path=None):
    # switch to evaluate mode
    preds = []
    gts = []
    with torch.no_grad():
        for i, sampled_batch in enumerate(tqdm(test_loader)):
            names, img, msk = sampled_batch['name'], sampled_batch['image'], sampled_batch['label'].long()
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = net(img)
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            score = torch.softmax(out, dim=1)
            out = torch.argmax(score, dim=1).cpu().detach().numpy()
            preds.append(out)
            if save_result:
                save_imgs(img, msk, out, score[0].cpu().detach().numpy(), i, test_save_path, dataset, threshold)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        # log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
        log_info = f'test of best model, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)

        single_metric = calculate_metric_percase(y_pre == 1, y_true == 1)
        print(single_metric)

    return [accuracy, sensitivity, specificity, f1_or_dsc, miou]
