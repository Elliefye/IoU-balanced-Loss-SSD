import sys

import numpy as np
from tqdm.autonotebook import tqdm
import torch
import model.eval as e


def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler):
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    scheduler.step()
    for i, (img, _, gloc, glabel) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold):
    model.eval()
    detections = []
    category_ids = [1, 2, 3]
    gt = []
    height, width = 300, 300

    for nbatch, (img, img_id, bbox, label) in enumerate(test_loader):

        for i in range(len(img_id)):
            try:
                # gt.append([img_id[i], bbox[i][0] * width, bbox[i][1] * height,
                #            (bbox[i][2] - bbox[i][0]) * width, (bbox[i][3] - bbox[i][1]) * height, label[i][0]])
                gt.append([img_id[i], round(float(bbox[i][0][0]) * 300), round(float(bbox[i][0][1]) * 300),
                           round(float(bbox[i][0][2]) * 300), round(float(bbox[i][0][3]) * 300), int(label[i][0])])
            except IndexError:
                # batch was smaller than expected
                pass
                # print('i = ' + str(i), len(img_id))

        print("Parsing batch: {}/{}".format(nbatch + 1, len(test_loader)))
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                detections_loc = []
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    # imgid, xmin, ymin, xmax, ymax, prob, label
                    # detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                    #                    (loc_[3] - loc_[1]) * height, prob_,
                    #                    category_ids[label_ - 1]])
                    detections_loc.append([img_id[idx], round(loc_[0] * 300), round(loc_[1] * 300), round(loc_[2] * 300),
                                       round(loc_[3] * 300), float(prob_), int(label_)])
                detections.append(detections_loc)

        # if nbatch >= 4:
        #     break

    detections = np.array(detections)
    gt = np.array(gt)

    e.evaluate_model(detections, gt)
    # print(detections[0])
    # print(gt[0].shape[0])

    # coco_eval = COCOeval(test_loader.dataset.coco.loadRes(), test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    #
    # writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)
