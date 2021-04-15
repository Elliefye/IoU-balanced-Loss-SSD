import torch
import torchvision.transforms as transforms
from model.utils import Encoder


class SimpleTransformer(object):
    def __init__(self, dboxes, eval=True):
        self.eval = eval
        self.size = (300, 300)  # only support 300x300 ssd
        self.dboxes = dboxes
        self.encoder = Encoder(self.dboxes)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
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

    def __call__(self, img, bbox=None, label=None, max_num=200):
        if not self.eval:
            bbox_out = torch.zeros(max_num, 4)
            label_out = torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bbox.size(0), :] = bbox
            label_out[:label.size(0)] = label
            return self.trans_eval(img), bbox_out, label_out

        img = self.img_trans(img).contiguous()
        bbox, label = self.encoder.encode(bbox, label)

        return img, bbox, label
