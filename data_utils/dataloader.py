import os
import pandas as pd
import numpy as np
import torch
import scipy.misc as m
import scipy.io as sio

from torch.utils.data import Dataset

import utils
from utils import recursive_glob
from PIL import Image
from .augmentations import *
import re
import ora


# dakle trenutno radim implementaciju u kojoj odmah slažem sve segmentacijske mape zubiju u jednu slikun sa cca 40 kanala
# to radim po uzoru na ovo: https://towardsdatascience.com/semantic-hand-segmentation-using-pytorch-3e7a0a0386fa
# slike nam moraju biti u one hot obliku prilikom ulaska u model
# a sad, hoću li to ostvariti u dataloaderu ili negdje drugo je moja stvar
class OrtoDataset(Dataset):
    split_ratio = [0.85, 0.15]  # podjela na label/unlabel ili train/val?

    # po novom, ratio označava omjer neoznačenih slika u odnosu na označene
    # dakle ako je 1, jednak je broj označenih i neoznačenih slika
    def __init__(self, ortopanograms_classes, root_path, name='label', ratio=1, transformation=None,
                 augmentation=None):  # ratio je valjda koliko od training slika da uzme za "neoznačene"?
        super(OrtoDataset, self).__init__()
        self.root_path = root_path  # dakle root, trenutni direktorij, podaci se nalaze u ../data mislim
        self.ratio = ratio
        self.name = name
        assert transformation is not None, 'transformation must be provided, give None'
        self.transformation = transformation
        self.augmentation = augmentation
        self.lower_teeth = ["38", "37", "36", "35", "34", "33", "32", "31", "48", "47", "46", "45", "44", "43", "42",
                            "41"]
        self.upper_teeth = ["11", "12", "13", "14", "15", "16", "17", "18", "21", "22", "23", "24", "25", "26", "27",
                            "28"]
        self.teeth_class_map = {"11": 1, "12": 2, "13": 3, "14": 4, "15": 5, "16": 6, "17": 7, "18": 8, "21": 9,
                                "22": 10, "23": 11, "24": 12, "25": 13, "26": 14, "27": 15, "28": 16,
                                "31": 17, "32": 18, "33": 19, "34": 20, "35": 21, "36": 22, "37": 23, "38": 24,
                                "41": 25, "42": 26, "43": 27, "44": 28, "45": 29, "46": 30, "47": 31, "48": 32}
        assert name in ('label', 'unlabel', 'val',
                        'test'), 'dataset name should be restricted in "label", "unlabel", "test" and "val", given %s' % name
        # assert 0 <= ratio <= 1, 'the ratio between "labeled" and "unlabeled" should be between 0 and 1, given %.1f' % ratio
        np.random.seed(1)  ### Because of this we are not getting repeated images for labelled and unlabelled data
        self.ortopanograms_classes = ortopanograms_classes

        if self.name != 'test':
            # ovdje samo radimo tablice, u tablici train_imgs su pathovi svih train slika, a u tablici val_imgs su pathovi svih val slika
            train_imgs = pd.read_table(os.path.join(self.root_path, 'imagelists', 'train.txt')).values.reshape(-1)
            val_imgs = pd.read_table(os.path.join(self.root_path, 'imagelists', 'val.txt')).values.reshape(-1)
            unlabeled_list = pd.read_table(os.path.join(self.root_path, 'imagelists', 'unlabeled.txt')).values.reshape(
                -1)

            # ovaj dio koda će možda trebati malo izmijeniti jednom kada ćemo imati neoznačene slike
            labeled_imgs = np.random.choice(train_imgs, size=int(train_imgs.__len__()), replace=False)
            labeled_imgs = list(labeled_imgs)

            unlabeled_imgs = np.random.choice(unlabeled_list, size=int(ratio * train_imgs.__len__()), replace=False)
            unlabeled_imgs = list(unlabeled_imgs)

            current_ratio = len(unlabeled_imgs) / len(labeled_imgs)

            # čini se da ovdje jednostavno uzmemo sve slike koje su nam dostupne, te neke stavimo da su nam labeled, neke ne
            # moguće da jednom kad ćeš actually imati dodatne neoznačene slike ćeš ovdje morati promijeniti kod
            ### Now here we equalize the lengths of labelled and unlabelled imgs by just repeating up some images
            if self.ratio > 1:
                # ako je omjer veći od 1, to znači da ima više neoznačenih slika nego označenih, te treba dodati još označenih slika

                new_imgs = np.random.choice(train_imgs, size=int(len(unlabeled_imgs) - len(labeled_imgs)),
                                            replace=False)
                new_list = list(new_imgs)
                labeled_imgs += new_list

                # new_ratio = round((self.ratio/(1-self.ratio + 1e-6)), 1)
                # excess_ratio = new_ratio - 1
                # new_list_1 = unlabeled_imgs * int(excess_ratio)
                # new_list_2 = list(np.random.choice(np.array(unlabeled_imgs), size=int((excess_ratio - int(excess_ratio))*unlabeled_imgs.__len__()), replace=False))
                # unlabeled_imgs += (new_list_1 + new_list_2)
            elif self.ratio < 1:
                new_imgs = np.random.choice(unlabeled_imgs, size=int(len(labeled_imgs) - len(unlabeled_imgs)),
                                            replace=False)
                new_list = list(new_imgs)
                unlabeled_imgs += new_list
                # new_ratio = round(((1-self.ratio)/(self.ratio + 1e-6)), 1)
                # excess_ratio = new_ratio - 1
                # new_list_1 = labeled_imgs * int(excess_ratio)
                # new_list_2 = list(np.random.choice(np.array(labeled_imgs), size=int((excess_ratio - int(excess_ratio))*labeled_imgs.__len__()), replace=False))
                # labeled_imgs += (new_list_1 + new_list_2)

        if self.name == 'test':
            test_imgs = pd.read_table(os.path.join(self.root_path, 'imagelists', 'test.txt')).values.reshape(-1)
            # test_imgs = np.array(test_imgs)
        if self.name == 'production':
            production_imgs = pd.read_table(os.path.join(self.root_path, 'imagelists', 'production.txt')).values.reshape(-1)

        # mislim da nam self.imgs onda postane popis slika s kojima radimo, bez nastavka .ora
        if self.name == 'label':
            self.imgs = labeled_imgs
        elif self.name == 'unlabel':
            self.imgs = unlabeled_imgs
        elif self.name == 'val':
            self.imgs = val_imgs
        elif self.name == 'test':
            self.imgs = test_imgs
        elif self.name == 'production':
            self.imgs = production_imgs
        else:
            raise ('{} not defined'.format(self.name))

        self.gts = self.imgs

    # This method loads a single (input, label) pair from the disk and performs whichever preprocessing the data requires. Pod label se valjda misli na ground truth?
    def __getitem__(self, index):
        #ovo je neka buduća ideja, ukoliko bi se kod ikad želio pustiti u "produkciju", onda
        #bi samo primao neoznačene slike, bez ground trutha, bez ičega
        if self.name == 'production':
            img_path = os.path.join(self.root_path, 'production', self.imgs[
                index] + '.png')  # pošto se radi o test datasetu, učitavamo samo .jpg, odnosno sliku, bez njene segmentacijske maske

            img = Image.open(img_path)  # .convert('RGB')

            if self.augmentation is not None:
                img = self.augmentation(img)

            if self.transformation:
                img = self.transformation['img'](img)

            # print("Vrijednosti slike sa RGB: " + np.unique(img.numpy()))

            return img, self.imgs[index]
        elif self.name == 'unlabel':
            img_path = os.path.join(self.root_path, 'unlabeled/data_store',
                                    self.imgs[index] + '.jpg')

            img = Image.open(img_path).convert('RGB')

            if self.augmentation is not None:
                img = self.augmentation(img)

            if self.transformation:
                img = self.transformation['img'](img)

            f = open("./logs/log.txt", "a")
            f.write("Koristim neoznacenu sliku: " + self.imgs[index] + "\n")
            f.close()

            # print("Vrijednosti slike sa RGB: " + np.unique(img.numpy()))
            return img, self.imgs[index]
        #dakle, ako se radi o training, validation ili testing
        #validation i testing su praktičiki identičan kod, samo što su različiti skupovi podataka
        #validation se koristi tijekom treniranja za donošenje odluke o updatanju modela
        #dok se training koristi nakon završetka treniranja modela, za procjenu uspješnosti modela
        else:
            ora_path = os.path.join(self.root_path, 'training',
                                    # u mapi /training nalaze se i training i validation slike, uzimaju se prema onom što piše u traing.txt, odnosno validation.txt
                                    self.imgs[index] + '.ora')  # jedno je valjda slika

            ora_image = ora.read_ora(ora_path)
            # učitavamo ortopanogram
            img = (ora_image['root']['childs'][0]['raster']).convert('RGB')
            # ova petlja je za svaki slučaj, ukoliko iz nekog razloga ora plugin sliku 000.png ne učita prvu
            # ako se ispostavi da uvijek 000.png čita prvu, možeš maknuti ovu petlju
            for i in range(len(ora_image['root']['childs'])):
                if (ora_image['root']['childs'][i]['src'] == 'data/000.png'):
                    img = (ora_image['root']['childs'][i]['raster']).convert('RGB')
                    break

            # img = Image.open(img_path).convert('RGB')

            # učitavanje ground trutha
            # Ex je oznaka za prazan zub
            # zadnja slika je uvijek pozadina, i nju treba učitati
            # dakle sad treba učitati samo slike koje ili nemaju slova u sebi, ili imaju samo Ex

            # potencijalna greška ukoliko background nije uvijek zadnji
            gt = ora_image['root']['childs'][-1]['raster'].convert(
                'L')  # ovo je da je spremi kao polycrhome, P znaci nesto drugo
            gt = self.transformation['gt'](gt)
            gt[gt != 0] = 1  # sve vrijednosti koje nisu 0 (pozadina) mijenjamo sa 1

            for i in range(len(ora_image['root']['childs'])):
                name = ora_image['root']['childs'][i]['name']
                # spajamo samo segmentacijske mape zubiju te pozadinu u jednu sliku/tensor -> one ili imaju duljinu imena 2 (npr. 17) ili imaju oblik poput 17 #1
                if (self.ortopanograms_classes == 2):
                    # pozadina, zubi
                    if (len(name) == 2 or ((len(name) == 5 and "#" in name))):
                        tmp_gt = ora_image['root']['childs'][i]['raster'].convert('L')
                        tmp_gt = self.transformation['gt'](tmp_gt)
                        tmp_gt[tmp_gt != 0] = 1

                        gt = gt | tmp_gt
                        # koristi or operaciju nad tenzorima da ih spojiš u jedan, ioanko su svi samo 0 ili jedan
                        # https://discuss.pytorch.org/t/combine-2-channels-of-an-image/75628/3
                        # kasnije kad ćeš imati više slika koristiti ćeš funkciju Relabel
                # pozadina, gornji zubi, donji zubi
                elif (self.ortopanograms_classes == 3):
                    if (len(name) == 2 or ((len(name)) == 5 and "#" in name)):
                        name = name[0:2]
                        tmp_gt = ora_image['root']['childs'][i]['raster'].convert('L')
                        tmp_gt = self.transformation['gt'](tmp_gt)

                        if (name in self.lower_teeth):
                            tmp_gt[tmp_gt != 0] = 1
                        elif (name in self.upper_teeth):
                            tmp_gt[tmp_gt != 0] = 2
                        else:
                            raise RuntimeError("Error! Unexpected tooth value - " + name)

                        gt = gt | tmp_gt #možda ovo zbraja :)

                        #dap dap dap ovo (|) ti zbraja
                        #to nije problem kad si imao samo 0 i 1 ali sad je
                        #i kad imaš preklapanje između piksela koji čine klasu 1 i 2
                        #on ti ih zbroji
                        #ok dogovor - ovo napiši u radu!
                        #kad imaš preklapanje, pridjeljuješ to gornjim zubima, dakle 2
                        #print("Vrijednosti gt-a iz dijela za 3 klase: " + str(np.unique(gt.numpy())))
                        # na kraju imamo jednokanalni tenzor, u kojem imamo nule na mjestima gdje su pikseli pozadine, 1 gdje su pikseli donjih zuba, te 2 gdje su pikseli gornjih zuba
                        # a mislim da će make_one_hot to pretvoriti u 3 kanalni tenzor
                        gt[gt == 3] = 2
                # pozadina, svaki zub je klasa za sebe
                elif (self.ortopanograms_classes == 33):
                    if (len(name) == 2 or ((len(name)) == 5 and "#" in name)):
                        name = name[0:2]
                        tmp_gt = ora_image['root']['childs'][i]['raster'].convert('L')
                        tmp_gt = self.transformation['gt'](tmp_gt)

                        tmp_gt[tmp_gt != 0] = self.teeth_class_map[name]
                        gt = gt | tmp_gt

            # ove 3 linije koristi ako hoce� provjeriti je li ucitava gt slike dobro, tj. spaja li ih dobro u jednu sliku
            new_img = gt.detach().squeeze().cpu().numpy()
            new_img = utils.colorize_mask(new_img, "ortopanograms", self.ortopanograms_classes)
            new_img.save(os.path.join("original_gt/" + self.imgs[index] + '.png'))
            #print("Spremam sliku " + self.imgs[index] + ".png")

            # ovdje ti može baciti grešku, jer će gt u ovom trenutku već biti tenzor
            if self.augmentation is not None:
                img, gt = self.augmentation(img, gt)

            if self.transformation:
                img = self.transformation['img'](img)
                # gt = self.transformation['gt'](gt) #mičemo ovu transformaciju jer smo je praktički gore več napravili

            # print("Vrijednosti slike sa RGB: " + str(np.unique(img.numpy())))
            print("Vrijednosti gt-a iz dataloadera: " + str(np.unique(gt.numpy())))
            return img, gt, self.imgs[index]

    # returns an integer describing the full length of the dataset
    def __len__(self):
        return len(self.imgs)

        # labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
        # labeled_imgs = list(labeled_imgs)
        #
        # unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

        # pošto se čini da punih označenih slika imaš dosta malo, možda će ti ovako nešto trebati?
        # podijela na označene i neoznačene
        # također, neku podijelu na test/train/validation imaš u metadata tablicama


class VOCDataset(Dataset):
    '''
    We assume that there will be txt to note all the image names


    color map:
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
    12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor,
    21=boundaries(self-defined)

    Also it will return an image and ground truth as it is present in the image form, so ground truth won't
    be in one-hot form and rather would be a 2D tensor. To convert the labels to one-hot form in the training
    code we will be calling the function 'make_one_hot' function of utils.py

    '''

    def __init__(self, root_path, name='label', ratio=0.5, transformation=None, augmentation=None):
        super(VOCDataset, self).__init__()
        self.root_path = root_path
        self.ratio = ratio
        self.name = name
        assert transformation is not None, 'transformation must be provided, give None'
        self.transformation = transformation
        self.augmentation = augmentation
        assert name in ('label', 'unlabel', 'val',
                        'test'), 'dataset name should be restricted in "label", "unlabel", "test" and "val", given %s' % name
        assert 0 <= ratio <= 1, 'the ratio between "labeled" and "unlabeled" should be between 0 and 1, given %.1f' % ratio
        np.random.seed(1)  ### Because of this we are not getting repeated images for labelled and unlabelled data

        if self.name != 'test':
            train_imgs = pd.read_table(
                os.path.join(self.root_path, 'ImageSets/Segmentation', 'trainvalAug.txt')).values.reshape(-1)

            val_imgs = pd.read_table(os.path.join(self.root_path, 'ImageSets/Segmentation', 'val.txt')).values.reshape(
                -1)

            # trebati će možda napraviti ovako neku podlijedu na train/val?

            # lol primijeti ovdje, ovdje on prvo učita skup svih slika za treniranje
            # a onda iz njih uzima da su mu neke slike označene neke ne
            # dakle trenutno kod i je napravljen za situaciju kad su ti sve slike označene, a ako kasnije budeš htio ubaciti neke prave neoznačene slike,
            # to ćeš morati sam dodati
            labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
            labeled_imgs = list(labeled_imgs)

            unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

            ### Now here we equalize the lengths of labelled and unlabelled imgs by just repeating up some images
            if self.ratio > 0.5:
                new_ratio = round((self.ratio / (1 - self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = unlabeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(unlabeled_imgs), size=int(
                    (excess_ratio - int(excess_ratio)) * unlabeled_imgs.__len__()), replace=False))
                unlabeled_imgs += (new_list_1 + new_list_2)
            elif self.ratio < 0.5:
                new_ratio = round(((1 - self.ratio) / (self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = labeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(labeled_imgs), size=int(
                    (excess_ratio - int(excess_ratio)) * labeled_imgs.__len__()), replace=False))
                labeled_imgs += (new_list_1 + new_list_2)

        if self.name == 'test':
            test_imgs = pd.read_table(
                os.path.join(self.root_path, 'ImageSets/Segmentation', 'test.txt')).values.reshape(-1)
            # test_imgs = np.array(test_imgs)

        if self.name == 'label':
            self.imgs = labeled_imgs
        elif self.name == 'unlabel':
            self.imgs = unlabeled_imgs
        elif self.name == 'val':
            self.imgs = val_imgs
        elif self.name == 'test':
            self.imgs = test_imgs
        else:
            raise ('{} not defined'.format(self.name))

        self.gts = self.imgs

    # This method loads a single (input, label) pair from the disk and performs whichever preprocessing the data requires. Pod label se valjda misli na ground truth?
    def __getitem__(self, index):
        if self.name == 'test':
            img_path = os.path.join(self.root_path, 'JPEGImages', self.imgs[
                index] + '.jpg')  # pošto se radi o test datasetu, učitavamo samo .jpg, odnosno sliku, bez njene segmentacijske maske

            img = Image.open(img_path).convert('RGB')

            if self.augmentation is not None:
                img = self.augmentation(img)

            if self.transformation:
                img = self.transformation['img'](img)

            return img, self.imgs[index]

        else:
            img_path = os.path.join(self.root_path, 'JPEGImages', self.imgs[index] + '.jpg')  # jedno je valjda slika
            gt_path = os.path.join(self.root_path, 'SegmentationClassAug',
                                   self.gts[index] + '.png')  # drugo je ručno označena maska
            # ja mislim da bi ja ovdje onda učitao svaih 30 i nešto slojeva i spojio ih u jednu sliku

            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path)  # .convert('P')

            if self.augmentation is not None:
                img, gt = self.augmentation(img, gt)

            if self.transformation:
                img = self.transformation['img'](img)
                gt = self.transformation['gt'](gt)

            return img, gt, self.imgs[index]

    # returns an integer describing the full length of the dataset
    def __len__(self):
        return len(self.imgs)


class CityscapesDataset(Dataset):

    def __init__(
            self,
            root_path,
            name="train",
            ratio=0.5,
            transformation=False,
            augmentation=None
    ):
        self.root = root_path
        self.name = name
        assert transformation is not None, 'transformation must be provided, give None'
        self.transformation = transformation
        self.augmentation = augmentation
        self.n_classes = 20
        self.ratio = ratio
        self.files = {}

        assert name in ('label', 'unlabel', 'val',
                        'test'), 'dataset name should be restricted in "label", "unlabel", "test" and "val", given %s' % name

        if self.name != 'test':
            self.images_base = os.path.join(self.root, "leftImg8bit")
            self.annotations_base = os.path.join(  # ovdje se spominju anotacije?
                self.root, "gtFine", 'trainval'
            )
        else:
            self.images_base = os.path.join(self.root, "leftImg8bit", 'test')
            # self.annotations_base = os.path.join(
            #     self.root, "gtFine", 'test'
            # )

        np.random.seed(1)

        if self.name != 'test':
            train_imgs = recursive_glob(rootdir=os.path.join(self.images_base, 'train'), suffix=".png")
            train_imgs = np.array(train_imgs)

            val_imgs = recursive_glob(rootdir=os.path.join(self.images_base, 'val'), suffix=".png")

            labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
            labeled_imgs = list(labeled_imgs)

            unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

            ### Now here we equalize the lengths of labelled and unlabelled imgs by just repeating up some images
            if self.ratio > 0.5:
                new_ratio = round((self.ratio / (1 - self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = unlabeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(unlabeled_imgs), size=int(
                    (excess_ratio - int(excess_ratio)) * unlabeled_imgs.__len__()), replace=False))
                unlabeled_imgs += (new_list_1 + new_list_2)
            elif self.ratio < 0.5:
                new_ratio = round(((1 - self.ratio) / (self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = labeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(labeled_imgs), size=int(
                    (excess_ratio - int(excess_ratio)) * labeled_imgs.__len__()), replace=False))
                labeled_imgs += (new_list_1 + new_list_2)

        else:
            test_imgs = recursive_glob(rootdir=self.images_base, suffix=".png")

        if (self.name == 'label'):
            self.files[name] = list(labeled_imgs)
        elif (self.name == 'unlabel'):
            self.files[name] = list(unlabeled_imgs)
        elif (self.name == 'val'):
            self.files[name] = list(val_imgs)
        elif (self.name == 'test'):
            self.files[name] = list(test_imgs)

        '''
        This pattern for the various classes has been borrowed from the official repo for Cityscapes dataset
        You can see it here: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        '''
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "unlabelled"
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[name]:
            raise Exception(
                "No files for name=[%s] found in %s" % (name, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[name]), name))

    def __len__(self):
        """__len__"""
        return len(self.files[self.name])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        if self.name == 'test':
            img_path = self.files[self.name][index].rstrip()

            img = Image.open(img_path).convert('RGB')

            if self.augmentation is not None:
                img = self.augmentation(img)

            if self.transformation:
                img = self.transformation['img'](img)

            return img, re.sub(r'.*/', '', img_path[34:]).rstrip(
                '.png')  ### These numbers have been hard coded so as to get a suitable name for the model

        else:
            img_path = self.files[self.name][index].rstrip()
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )

            img = Image.open(img_path).convert('RGB')
            lbl = Image.open(lbl_path)

            if self.augmentation is not None:
                img, lbl = self.augmentation(img, lbl)

            if self.transformation:
                img = self.transformation['img'](img)
                lbl = self.transformation['gt'](lbl)

            lbl = self.encode_segmap(lbl)  # zanimljivo

            return img, lbl, re.sub(r'.*/', '', img_path[38:]).rstrip(
                '.png')  ### These numbers have been hard coded so as to get a suitable name for the model

    def encode_segmap(self, mask):  # moguće da se ovdje segmentacijske karte spajaju u jednu veliku sliku?
        # Put all void classes to ignore index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        mask[mask == self.ignore_index] = 19  ### Just a mapping between the two color values
        return mask


class ACDCDataset(Dataset):  # ovaj ti je možda još i najsličniji tvom
    '''
    The dataloader for ACDC dataset
    '''

    split_ratio = [0.85, 0.15]
    '''
    this split ratio is for the train (including the labeled and unlabeled) and the val dataset
    '''

    def __init__(self, root_path, name='label', ratio=0.5, transformation=None, augmentation=None):
        super(ACDCDataset, self).__init__()
        self.root = root_path
        self.name = name
        assert transformation is not None, 'transformation must be provided, give None'
        self.transformation = transformation
        self.augmentation = augmentation
        self.ratio = ratio
        self.files = {}

        if self.name != 'test':
            self.images_base = os.path.join(self.root, 'training')
            self.annotations_base = os.path.join(self.root, 'training_gt')
        else:
            self.images_base = os.path.join(self.root, 'testing')
            # self.annotations_base = os.path.join(
            #     self.root, "gtFine", 'test'
            # )

        np.random.seed(1)

        if self.name != 'test':
            total_imgs = os.listdir(self.images_base)
            total_imgs = np.array(total_imgs)

            train_imgs = np.random.choice(total_imgs, size=int(self.__class__.split_ratio[0] * total_imgs.__len__()),
                                          replace=False)
            val_imgs = [x for x in total_imgs if x not in train_imgs]

            labeled_imgs = np.random.choice(train_imgs, size=int(self.ratio * train_imgs.__len__()), replace=False)
            labeled_imgs = list(labeled_imgs)

            unlabeled_imgs = [x for x in train_imgs if x not in labeled_imgs]

            ### Now here we equalize the lengths of labelled and unlabelled imgs by just repeating up some images
            if self.ratio > 0.5:
                new_ratio = round((self.ratio / (1 - self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = unlabeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(unlabeled_imgs), size=int(
                    (excess_ratio - int(excess_ratio)) * unlabeled_imgs.__len__()), replace=False))
                unlabeled_imgs += (new_list_1 + new_list_2)
            elif self.ratio < 0.5:
                new_ratio = round(((1 - self.ratio) / (self.ratio + 1e-6)), 1)
                excess_ratio = new_ratio - 1
                new_list_1 = labeled_imgs * int(excess_ratio)
                new_list_2 = list(np.random.choice(np.array(labeled_imgs), size=int(
                    (excess_ratio - int(excess_ratio)) * labeled_imgs.__len__()), replace=False))
                labeled_imgs += (new_list_1 + new_list_2)

        else:
            test_imgs = os.listdir(self.images_base)

        if (self.name == 'label'):
            self.files[name] = list(labeled_imgs)
        elif (self.name == 'unlabel'):
            self.files[name] = list(unlabeled_imgs)
        elif (self.name == 'val'):
            self.files[name] = list(val_imgs)
        elif (self.name == 'test'):
            self.files[name] = list(test_imgs)

        if not self.files[name]:
            raise Exception(
                "No files for name=[%s] found in %s" % (name, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[name]), name))

    def __len__(self):
        """__len__"""
        return len(self.files[self.name])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        if self.name == 'test':
            img_path = os.path.join(self.images_base, self.files[self.name][index])

            img = Image.open(img_path)

            if self.augmentation is not None:
                img = self.augmentation(img)

            if self.transformation:
                img = self.transformation['img'](img)

            return img, self.files[self.name][index].rstrip('.jpg')

        else:
            img_path = os.path.join(self.images_base, self.files[self.name][index])
            lbl_path = os.path.join(self.annotations_base, self.files[self.name][index].rstrip('.jpg') + '.png')

            img = Image.open(img_path)
            lbl = Image.open(lbl_path)

            if self.augmentation is not None:
                img, lbl = self.augmentation(img, lbl)

            if self.transformation:
                img = self.transformation['img'](img)
                lbl = self.transformation['gt'](lbl)

            return img, lbl, self.files[self.name][index].rstrip('.jpg')  # ovo zadnje je samo ime
