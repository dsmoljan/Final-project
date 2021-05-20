import os
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from PIL import Image
from arch import define_Gen
from data_utils import VOCDataset, CityscapesDataset, ACDCDataset, get_transformation, OrtoDataset

root = '../data/VOC2012test/VOC2012'
root_cityscapes = '../data/Cityscape'
root_acdc = '../data/ACDC'
root_ortopanograms = '../../data/Ortopanograms'


def test(args):
    ### For selecting the number of channels
    if args.dataset == 'voc2012':
        n_channels = 21
    elif args.dataset == 'cityscapes':
        n_channels = 20
    elif args.dataset == 'acdc':
        n_channels = 4
    elif args.dataset == 'ortopanograms':
        n_channels = 2
    elif args.dataset == 'ortopanograms_test_output':
        n_channels = 2

    running_metrics_test = utils.runningScore(n_channels, args.dataset)

    transform = get_transformation((args.crop_height, args.crop_width), resize=True, dataset=args.dataset)

    ## let the choice of dataset configurable
    if args.dataset == 'voc2012':
        test_set = VOCDataset(root_path=root, name='test', ratio=0.5, transformation=transform, augmentation=None)
    elif args.dataset == 'cityscapes':
        test_set = CityscapesDataset(root_path=root_cityscapes, name='test', ratio=0.5, transformation=transform,
                                     augmentation=None)
    elif args.dataset == 'acdc':
        test_set = ACDCDataset(root_path=root_acdc, name='test', ratio=0.5, transformation=transform, augmentation=None)
    elif args.dataset == 'ortopanograms':
        test_set = OrtoDataset(args.ortopanograms_classes, root_path=root_ortopanograms, name='test', ratio=0.5, transformation=transform,
                               augmentation=None)
    elif args.dataset == 'ortopanograms_test_output':
        test_set = OrtoDataset(root_path=root_ortopanograms, name='test', ratio=1, transformation=transform,
                               augmentation=None)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    Gsi = define_Gen(input_nc=3, output_nc=n_channels, ngf=args.ngf, netG='deeplab',
                     norm=args.norm, use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)


    ### activation_softmax
    activation_softmax = nn.Softmax2d()

    if (args.model == 'supervised_model'):

        print("Running test for supervised model")

        interp = nn.Upsample((args.crop_height, args.crop_width), mode='bilinear', align_corners=True)


        ### loading the checkpoint
        try:
            ckpt = utils.load_checkpoint('%s/latest_supervised_model.ckpt' % (args.checkpoint_dir))
            Gsi.load_state_dict(ckpt['Gsi'])

        except:
            print(' [*] No checkpoint!')

        ### run
        Gsi.eval()
        for i, (test_img, test_gt, image_name) in enumerate(test_loader):

            test_img, test_gt = utils.cuda([test_img,test_gt], args.gpu_ids)
            seg_map = Gsi(test_img)
            seg_map = interp(seg_map)  # samo povecavamo sliku, slika koju dobijemo iz GSI je onak 20xx40
            seg_map = activation_softmax(seg_map)

            prediction = seg_map.data.max(1)[1].squeeze_(1).squeeze_(
                0).cpu().numpy()  ### To convert from 22 --> 1 channel
            test_gt = test_gt.squeeze().data.cpu().numpy()
            running_metrics_test.update(test_gt, prediction)

            # outputs = self.Gsi(val_img)
            # outputs = self.interp_val(outputs)
            # outputs = self.activation_softmax(outputs)
            #
            # pred = outputs.data.max(1)[1].cpu().numpy()
            # gt = val_gt.squeeze().data.cpu().numpy()
            #
            # self.running_metrics_val.update(gt, pred)
            for j in range(prediction.shape[0]):
                new_img = prediction[j]  ### Taking a particular image from the batch
                new_img = utils.colorize_mask(new_img, args.dataset,
                                              args.ortopanograms_classes)  ### So as to convert it back to a paletted image

                ### Now the new_img is PIL.Image
                new_img.save(os.path.join(args.results_dir + '/supervised/' + image_name[j] + '.png'))

            print('Epoch-', str(i + 1), ' Done!')
        score, class_iou = running_metrics_test.get_scores()
        print("Test mIOU is: ", score["Mean IoU : \t"])

    elif (args.model == 'semisupervised_cycleGAN'):

        print("Running test for semisupervised model")

        interp = nn.Upsample((args.crop_height, args.crop_width), mode='bilinear', align_corners=True)

        ### loading the checkpoint
        try:
            ckpt = utils.load_checkpoint('%s/latest_supervised_model.ckpt' % (args.checkpoint_dir))
            Gsi.load_state_dict(ckpt['Gsi'])
            best_iou = ckpt['best_iou']

        except:
            print(' [*] No checkpoint!')

        ### run
        Gsi.eval()
        Gis.eval()

        for i, (test_img, test_gt, image_name) in enumerate(test_loader):
            test_img, test_gt = utils.cuda([test_img,test_gt], args.gpu_ids)
            seg_map = Gsi(test_img)
            seg_map = activation_softmax(seg_map)
            seg_map = interp(seg_map)  # samo povecavamo sliku, slika koju dobijemo iz GSI je onak 20xx40

            prediction = seg_map.data.max(1)[1].squeeze_(1).squeeze_(
                0).cpu().numpy()  ### To convert from 22 --> 1 channel
            test_gt = test_gt.squeeze().data.cpu().numpy()
            running_metrics_test.update(test_gt, prediction)
            #ovdje samo iteriramo po batchevima, to je shape[0]
            #ukoliko ćeš htjeti za više klasa, morati ćeš prepraviti kod
            for j in range(prediction.shape[0]):
                new_img = prediction[j]  ### Taking a particular image from the batch

                new_img = utils.colorize_mask(new_img, args.dataset,
                                              args.ortopanograms_classes)  ### So as to convert it back to a paletted image

                ### Now the new_img is PIL.Image
                new_img.save(os.path.join(args.results_dir + '/unsupervised/' + image_name[j] + '.png'))

            print('Epoch-', str(i + 1), ' Done!')

        score, class_iou = running_metrics_test.get_scores()
        print("Test mIOU is: ", score["Mean IoU : \t"])