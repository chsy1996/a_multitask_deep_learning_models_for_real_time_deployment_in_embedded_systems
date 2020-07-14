# Combine StanfordDrone and Dronedeploy datasets
import torch
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
from PIL import Image

from datasets.transform import sdd_transform, fcn_transform
import torchvision

sdd_labels = ('biker', 'pedestrian', 'skater', 'cart', 'car', 'bus')
sdd_labelmap = {k: v + 1 for v, k in enumerate(sdd_labels)}
sdd_labelmap['background'] = 0  # {[label_name]: [label_number]}
rev_sdd_labelmap = {v: k for k,v in sdd_labelmap.items()}

dd_classes = ['building', 'clutter', 'vegetation',
              'water', 'ground', 'car']

dd_colormap = [[75, 25, 230], [180, 30, 145],[75, 180, 60],
               [48, 130, 245], [255, 255, 255], [200, 130, 000]]

assert len(dd_classes) == len(dd_colormap)

def parse_annotations(annotation_path):
    """

    :param annotation_path: path to xml file in Annotation file.
    :return:
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in sdd_labelmap:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(sdd_labelmap[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


class MtlDataset(Dataset):
    """
    Combine stanforddrone and dronedeploy datasets
    """
    def __init__(self, data_folder, split, keep_difficult=False):

        super(MtlDataset, self).__init__()

        self.data_folder = data_folder

        self.split = split
        self.split = self.split.lower()
        assert self.split in {'train', 'val', 'test'}

        self.keep_difficult = keep_difficult


        # StanfordDrone dataset
        self.sdd_path = os.path.join(self.data_folder, 'StanfordDroneDataset/sdd')

        # ids of images
        with open(os.path.join(self.sdd_path, 'ImageSets/Main', split + '.txt')) as f:
            ids = f.read().splitlines()

        #
        self.sdd_images = list()
        self.sdd_objects = list()
        n_objects = 0

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotations(os.path.join(self.sdd_path, 'Annotations', id + '.xml'))
            if len(objects["boxes"]) == 0:  # filter out the picture that have no ground_truth boxes, to avoid error in find_intersection function
                continue
            n_objects += len(objects)
            self.sdd_objects.append(objects)
            self.sdd_images.append(os.path.join(self.sdd_path, 'JPEGImages', id + '.jpg'))

        assert len(self.sdd_objects) == len(self.sdd_images)

        print('-----------------------------------------------------------------------------------------------------')
        print('|| For StanfordDroneDataset, there are ', len(self.sdd_images),' ', split,
              ' images containing total of ', n_objects, ' objects. ||')


        # DroneDeploy dataset
        dd_path = os.path.join(self.data_folder, 'dataset-medium')
        with open(os.path.join(dd_path, split + '.txt')) as f:
            ids = f.read().splitlines()

        self.dd_images = list()
        self.dd_labels = list()
        for id in ids:
            self.dd_images.append(os.path.join(dd_path, 'image-chips', id))
            self.dd_labels.append(os.path.join(dd_path, 'label-chips', id))

        assert len(self.dd_images) == len(self.dd_labels)
        print('|| For DroneDeploy dataset, there are %d '%(len(self.dd_images)), split,
              ' images with corresponding %d'%(len(self.dd_labels)), ' labels.       ||')
        print('-----------------------------------------------------------------------------------------------------')


    def __getitem__(self, i):
        # sdd data
        # Read image
        sdd_image = Image.open(self.sdd_images[i], mode='r')
        sdd_image = sdd_image.convert('RGB')

        # Read objects in this sdd image(bounding boxes, labels, difficulties)
        sdd_objects = self.sdd_objects[i]
        sdd_boxes = torch.FloatTensor(sdd_objects['boxes']) # (n_objects, 4)
        sdd_labels = torch.LongTensor(sdd_objects['labels'])    # (n_objects)
        sdd_difficulties = torch.BoolTensor(sdd_objects['difficulties']) # (n_objects)

        if not self.keep_difficult:
            sdd_boxes = sdd_boxes[~sdd_difficulties]
            sdd_labels = sdd_labels[~sdd_difficulties]
            sdd_difficulties = sdd_difficulties[~sdd_difficulties]

        # Apply transforms
        sdd_image, sdd_boxes, sdd_labels, sdd_difficulties = \
            sdd_transform(sdd_image, sdd_boxes, sdd_labels, sdd_difficulties ,self.split.upper())

        # dronedeploy dataset
        while i >= len(self.dd_images): # note here is >=
            i -= len(self.dd_images)

        dd_image = Image.open(self.dd_images[i]).convert('RGB')
        dd_label = Image.open(self.dd_labels[i]).convert('RGB')

        dd_image, dd_label = fcn_transform(dd_image, dd_label, self.split.upper())


        return dd_image, dd_label, sdd_image, sdd_boxes, sdd_labels, sdd_difficulties

    def __len__(self):
        return len(self.sdd_images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the dataloader).

        This describes how to combine these tensors of different sizes. we use lists.

        Note: This need not be defined in this class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of dronedeploy images, a tensor of dronedeploy labels, a tensor of ssd images, lists of
                varying-size tensors of bounding boxes, labels and difficulties.
        """

        dd_images = list()
        dd_labels = list()
        sdd_images = list()
        sdd_boxes = list()
        sdd_labels = list()
        sdd_difficulties = list()

        for b in batch:
            dd_images.append(b[0])
            dd_labels.append(b[1])
            sdd_images.append(b[2])
            sdd_boxes.append((b[3]))
            sdd_labels.append(b[4])
            sdd_difficulties.append(b[5])

        dd_images = torch.stack(dd_images, dim=0)
        dd_labels = torch.stack(dd_labels, dim=0)

        sdd_images = torch.stack(sdd_images, dim=0)

        return dd_images, dd_labels, sdd_images, sdd_boxes, sdd_labels, sdd_difficulties

if __name__ == '__main__':

    print(sdd_labelmap)

    data_folder = '/home/chsy1996/Downloads/datasets'

    data = MtlDataset(data_folder=data_folder, split='train')
    dd_img, dd_lbl, img, boxes, lbl, diffi = data[11758]
    print(len(data),dd_img.shape, dd_lbl.shape, img.shape, boxes, lbl,diffi.shape)

    print(lbl)
