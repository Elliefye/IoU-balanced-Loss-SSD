import torch
import torchvision.transforms as transforms
from model.utils import Encoder
import random
from PIL import Image


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.random() < self.prob:
            bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]
            return img.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return img, bboxes


class SimpleTransformer(object):
    def __init__(self, dboxes, eval=False):
        self.eval = eval
        self.size = (300, 300)  # only support 300x300 ssd
        self.dboxes = dboxes
        self.encoder = Encoder(self.dboxes)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.hflip = RandomHorizontalFlip()
        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            transforms.ToTensor(),
            self.normalize
        ])
        self.trans_eval = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize
        ])

    def __call__(self, img, bboxes=None, labels=None, max_num=100):
        if self.eval:
            bbox_out = torch.zeros(max_num, 4)
            label_out = torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bboxes.size(0), :] = bboxes
            label_out[:labels.size(0)] = labels
            return self.trans_eval(img), bbox_out, label_out

        img, bboxes = self.hflip(img, bboxes)
        img = self.img_trans(img).contiguous()
        bboxes, labels = self.encoder.encode(bboxes, labels)

        return img, bboxes, labels
