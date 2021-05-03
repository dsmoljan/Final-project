import copy
import os
import shutil

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import models
from collections import namedtuple
from torchvision import transforms

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

cityscape_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153,
                      250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 0, 130, 180, 220, 20, 60,
                      255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

acdc_palette = [0, 0, 0, 128, 64, 128, 70, 70, 70, 250, 170, 30] #iz pocetnih 0,0,0 vidimo da je pozadina sigurno jedna od klasa u modelu

#0,0,0 -> pozadina
#255,0,0 -> zubi (crvena)
ortopanograms_palette = [0,0,0, 255, 255, 0] #ova paleta koristi se za prikaz RGB slike u tensorboardu prilikom treniranja modela

#ortopanograms_palette = [255,255,0] #ova paleta koristi se prilikom outputa modela, odnosno prilikom
#generiranja slika iz jednokanalnih tenzora, koji su izlaz modela u test nacinu rada

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

zero_pad = 256 * 3 - len(cityscape_palette)
for i in range(zero_pad):
    cityscape_palette.append(0)

zero_pad = 256 * 3 - len(acdc_palette)
for i in range(zero_pad):
    acdc_palette.append(0)

zero_pad = 256 * 3 - len(ortopanograms_palette)
for i in range(zero_pad):
    ortopanograms_palette.append(0)



def colorize_mask(mask, dataset):
    '''
    Used to convert the segmentation of one channel(mask) back to a paletted image
    '''
    # mask: numpy array of the mask
    assert dataset in ('voc2012', 'cityscapes', 'acdc','ortopanograms','ortopanograms_test_output')
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    if (dataset == 'voc2012'):
        new_mask.putpalette(palette)
    elif (dataset == 'cityscapes'):
        new_mask.putpalette(cityscape_palette)
    elif (dataset == 'acdc'):
        new_mask.putpalette(acdc_palette)
    elif (dataset == 'ortopanograms'):
        new_mask.putpalette(ortopanograms_palette)
    return new_mask

### To convert a paletted image to a tensor image of 3 dimension
### This is because a simple paletted image cannot be viewed with all the details
def PIL_to_tensor(img, dataset):
    '''
    Here img is of the type PIL.Image
    '''
    assert dataset in ('voc2012', 'cityscapes', 'acdc', 'ortopanograms')
    img_arr = np.array(img, dtype='float32')
    new_arr = np.zeros([3, img_arr.shape[0], img_arr.shape[1]], dtype='float32')

    if (dataset == 'voc2012'):
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                # new_arr[i, :, :] = img_arr
                index = int(img_arr[i, j]*3)
                new_arr[0, i, j] = palette[index]
                new_arr[1, i, j] = palette[index+1]
                new_arr[2, i, j] = palette[index+2]
    elif (dataset == 'cityscapes'):
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                # new_arr[i, :, :] = img_arr
                index = int(img_arr[i, j]*3)
                new_arr[0, i, j] = cityscape_palette[index]
                new_arr[1, i, j] = cityscape_palette[index+1]
                new_arr[2, i, j] = cityscape_palette[index+2]
    elif (dataset == 'acdc'):
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                # new_arr[i, :, :] = img_arr
                index = int(img_arr[i, j]*3)
                new_arr[0, i, j] = acdc_palette[index]
                new_arr[1, i, j] = acdc_palette[index+1]
                new_arr[2, i, j] = acdc_palette[index+2]
    elif (dataset == 'ortopanograms'):
        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                # new_arr[i, :, :] = img_arr
                index = int(img_arr[i, j]*3)
                new_arr[0, i, j] = ortopanograms_palette[index]
                new_arr[1, i, j] = ortopanograms_palette[index+1]
                new_arr[2, i, j] = ortopanograms_palette[index+2]

    
    return_tensor = torch.tensor(new_arr)

    return return_tensor

def smoothen_label(label, alpha, gpu_id):
    '''
    For smoothening of the classification labels
    
    labels : tensor having dimensrions: batch_size*21*H*W filled with zeroes and ones
    '''
    torch.manual_seed(0)
    try:
        smoothen_array = -1*alpha + torch.rand([label.shape[0], label.shape[1], label.shape[2], label.shape[3]]) * (2*alpha)
        smoothen_array = cuda(smoothen_array, gpu_id)
        label = label + smoothen_array
    except:
        smoothen_array = -1*alpha + torch.rand([label.shape[0], label.shape[1], label.shape[2], label.shape[3]]) * (2*alpha)
        label = label + smoothen_array

    return label

'''
To be used to apply gaussian noise in the input to the discriminator
'''
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = torch.zeros(x.size()).normal_() * scale
            x = x + sampled_noise
        return x

'''
This will be used for calculation of perceptual losses
'''
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

'''
The definition of the perceptual loss
'''
def perceptual_loss(x, y, gpu_ids):
    """
    Calculates the perceptual loss on the basis of the VGG network
    Parameters:
    x, y: the images between which perceptual loss is to be calculated
    """

    ### Considering the fact in this case x,y both are images in the range -1 to 1 and we need normal distribution
    ### before passing through VGG

    u = x*0.5 + 0.5
    v = y*0.5 + 0.5
    trans_mean = [0.485, 0.456, 0.406]
    trans_std = [0.229, 0.224, 0.225]

    for i in range(3):
        u[:, i, :, :] = u[:, i, :, :]*trans_std[i] + trans_mean[i]
        v[:, i, :, :] = v[:, i, :, :]*trans_std[i] + trans_mean[i]

    ### Now this is normal distribution

    vgg = Vgg16(requires_grad=False).cuda(gpu_ids[0])

    features_y = vgg(v)
    features_x = vgg(u)

    mse_loss = nn.MSELoss()
    loss = mse_loss(features_y.relu2_2, features_x.relu2_2)

    return loss


# To make directories
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


# To make cuda tensor
def cuda(xs, gpu_id):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda(int(gpu_id[0]))
        else:
            return [x.cuda(int(gpu_id[0])) for x in xs]
    return xs


# For Pytorch datasets loader
def create_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    mkdir(dirs.values())

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], 'Link'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                   os.path.join(dirs[key], 'Link'))

    return dirs


def get_traindata_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    return dirs


def get_testdata_link(dataset_dir):
    dirs = {}
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    return dirs


# To save the checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# To store 50 generated images in a pool and sample from it when it is full
# Shrivastava et al�s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

# mislim da ova metoda sluzi samo za labele (segmentacijske mape)
# ovdje ustvari dolazi do razdvajanja
def make_one_hot(labels, dataname, gpu_id):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. # shape [batch, channels, rows, columns] -> kako smo odjednom dosli do 1 kanala, kad su slike bile RGB?
        Each value is an integer representing correct classification. -> dakle, na neku foru ovdje je jedna slika (pusti sad N, to je samo batch size) je slika dimenzija WxH sa brojevima, u kojoj je svaki broj ispravna klasa
        probaj,probaj,probaj i ti to dobiti sa svojim podacima! ne zvuci nemoguce!
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    assert dataname in ('voc2012', 'cityscapes', 'acdc', 'ortopanograms'),'dataset name should be one of the following: \'voc2012\',given {}'.format(dataname)


    if dataname == 'voc2012':
        C = 21
    elif dataname == 'cityscapes':
        C = 20
    elif dataname == 'acdc':
        C = 4
    elif dataname == 'ortopanograms':
        C = 2
    else:
        raise NotImplementedError

    labels = labels.long()
    try:
        #ovdje radi one hot sliku koja ima isto batcheva kao originalna (to znaci labels.size(0), C kanala, isto redaka i stupaca kao i originalna (labels.size(2) i labels.size(3))
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_() #https://stackoverflow.com/questions/55565687/i-dont-understand-the-code-for-training-a-classifier-in-pytorch obja�njava �to znaci npr. labels(0)
        one_hot = cuda(one_hot, gpu_id)
    except:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target


# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py


class runningScore(object):
    def __init__(self, n_classes, dataset):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.dataset = dataset

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    #trebat ce dodati za ortopanograme!
    #ovdje se racuna MIoU
    #moguce da ce trebati prepraviti kod da se doda ba� specifican nacin izracuna za ortopanograme, ne svida mi se �to
    #stalno vraca 1.0
    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        if self.dataset == 'voc2012':
            iu = np.diag(hist[1:self.n_classes, 1:self.n_classes]) / (hist[1:self.n_classes, 1:self.n_classes].sum(axis=1) + hist[1:self.n_classes, 1:self.n_classes].sum(axis=0) - np.diag(hist[1:self.n_classes, 1:self.n_classes]))
        elif self.dataset == 'cityscapes':
            iu = np.diag(hist[0:self.n_classes-1, 0:self.n_classes-1]) / (hist[0:self.n_classes-1, 0:self.n_classes-1].sum(axis=1) + hist[0:self.n_classes-1, 0:self.n_classes-1].sum(axis=0) - np.diag(hist[0:self.n_classes-1, 0:self.n_classes-1]))
        elif self.dataset == 'acdc' or self.dataset == 'ortopanograms':
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        # freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        if self.dataset == 'voc2012' or self.dataset == 'cityscapes' or self.dataset == 'ortopanograms':
            cls_iu = dict(zip(range(self.n_classes-1), iu))
        elif self.dataset == 'acdc':
            cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
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


class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)

def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')
