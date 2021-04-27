import json
import numpy as np
import torch
from PIL import Image
from flask import Flask
from flask_restful import Api, Resource, reqparse
import requests
from io import BytesIO
from model.utils import generate_dboxes, Encoder
from model.ssd import SSD
from model.transform import SimpleTransformer

app = Flask(__name__)
app.config['DEBUG'] = True
api = Api(app)


class Model(Resource):
    def ensure_legal(self, xmin, ymin, xmax, ymax, width, height):
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > width:
            xmax = width
        if ymax > height:
            ymax = height
        return xmin, ymin, xmax, ymax

    def post(self):
        ret = []
        nms_threshold = 0.5
        cls_threshold = 0.3
        classes = ['Background', 'Knife', 'Horse', 'Human']
        parser = reqparse.RequestParser()
        parser.add_argument('img', required=True)
        args = parser.parse_args()

        imgurl = args['img']
        response = requests.get(imgurl, stream=True)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        width, height = img.size

        model = SSD()
        checkpoint = torch.load('C:\\Users\\Elina\\Desktop\\IoU-balanced-Loss-SSD\\trained_models\\SSD.pth')
        model.load_state_dict(checkpoint["model_state_dict"])
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        dboxes = generate_dboxes()
        transformer = SimpleTransformer(dboxes, eval=True)
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

            if len(loc) > 0:
                loc[:, 0::2] *= width
                loc[:, 1::2] *= height
                loc = loc.astype(np.int32)
                for box, lb, pr in zip(loc, label, prob):
                    category = classes[lb]
                    xmin, ymin, xmax, ymax = box
                    xmin, ymin, xmax, ymax = self.ensure_legal(xmin, ymin, xmax, ymax, width, height)
                    ret.append({'label': category, 'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]})

        return json.dumps(ret), 200


# Add URL endpoints
api.add_resource(Model, '/model')

if __name__ == '__main__':
    app.run()
