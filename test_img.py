import numpy as np
import torch
import os

from model.transform import SimpleTransformer
import cv2
from PIL import Image

from model.utils import generate_dboxes, Encoder, colors
from model.ssd import SSD

classes = ['Background', 'Knife', 'Horse', 'Human']
input_folder = 'Dataset_lim/validation/human_body/'
cls_threshold = 0.3
nms_threshold = 0.5
model_path = 'trained_models/SSD.pth'
output_path = 'predictions/'


def ensure_legal(xmin, ymin, xmax, ymax, width, height):
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > width:
        xmax = width
    if ymax > height:
        ymax = height
    return xmin, ymin, xmax, ymax


def test_one(path):
    model = SSD()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    dboxes = generate_dboxes()
    transformer = SimpleTransformer(dboxes, eval=True)
    img = Image.open(path).convert("RGB")
    img, _, _ = transformer(img, torch.zeros(4), torch.zeros(1))
    encoder = Encoder(dboxes)

    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        ploc, plabel = model(img.unsqueeze(dim=0))
        result = encoder.decode_batch(ploc, plabel, nms_threshold, 20)[0]
        loc, label, prob = [r.cpu().numpy() for r in result]
        best = np.argwhere(prob > cls_threshold).squeeze(axis=1)
        loc = loc[best]
        label = label[best]
        prob = prob[best]
        output_img = cv2.imread(path)
        if len(loc) > 0:
            height, width, _ = output_img.shape
            loc[:, 0::2] *= width
            loc[:, 1::2] *= height
            loc = loc.astype(np.int32)
            for box, lb, pr in zip(loc, label, prob):
                category = classes[lb]
                color = colors[lb]
                xmin, ymin, xmax, ymax = box
                xmin, ymin, xmax, ymax = ensure_legal(xmin, ymin, xmax, ymax, width, height)
                cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,
                              -1)
                cv2.putText(
                    output_img, category + " : %.2f" % pr,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
        output = output_path + "{}_prediction.jpg".format(os.path.splitext(os.path.split(path)[1])[0])
        cv2.imwrite(output, output_img)


if __name__ == "__main__":
    for image in os.listdir(input_folder):
        test_one(input_folder + image)
