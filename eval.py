from model.MtlModel import MtlModel
from model.MultiBoxLoss import MultiBoxLoss
from datasets.MtlDataset import MtlDataset
from torch.utils.data import DataLoader
from utils import *
import time
from metrics import *
from pprint import PrettyPrinter
import torch

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config
data_folder = '/home/chsy1996/Downloads/datasets/'
batch_size = 6
fcn_classes = 6
ssd_classes = 7
checkpoint = './checkpoint_mtl.pth.tar'
keep_difficult = True

# Load model from checkpoint
checkpoint = torch.load(checkpoint)
mtlmodel = checkpoint['model']
mtlmodel = mtlmodel.to(device)

# Switch to eval mode
mtlmodel.eval()

fcn_criterion = torch.nn.CrossEntropyLoss().to(device)
ssd_criterion = MultiBoxLoss(priors_cxcy=mtlmodel.priors_cxcy).to(device)

# prepare eval datasets
val_data = MtlDataset(data_folder=data_folder, split='val', keep_difficult=keep_difficult)
val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=val_data.collate_fn, shuffle=True)

def eval(valloader, model, fcn_criterion, ssd_criterion):
    """

    :param valloader:
    :param model:
    :param fcn_criterion:
    :param ssd_criterion:
    :return:
    """

    # fcn metrics: Acc, Mean IU,
    fcn_loss_meter = AverageMeter()
    fcn_acc_meter = AverageMeter()
    fcn_acc_cls_meter = AverageMeter()
    fcn_mean_iu_meter = AverageMeter()
    fcn_fwavacc_meter = AverageMeter()

    # ssd metrics: mAP
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list() # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    batch_time = AverageMeter()
    data_time = AverageMeter()
    start = time.time()

    with torch.no_grad():
        # Batches
        for i, data in enumerate(valloader):

            data_time.update(time.time() - start)

            fcn_imgs = data[0].to(device)
            fcn_lbls = data[1].to(device)
            ssd_imgs = data[2].to(device)   # (N, 3, 300, 300)
            ssd_boxes = data[3]
            ssd_lbls = data[4]
            ssd_difficulties = data[5]

            # fcn forward prop.
            fcn_out, _, _ = model(fcn_imgs)
            fcn_loss = fcn_criterion(fcn_out, fcn_lbls)
            fcn_loss_meter.update(fcn_loss.data)

            # ssd forward prop.
            _, predicted_locs, predicted_scores = model(ssd_imgs)

            ssd_loss = ssd_criterion(predicted_locs, predicted_scores, ssd_boxes, ssd_lbls)

            # Detect objects in SSD output.
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                      min_score=0.01, max_overlap=0.45,
                                                                                      top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparison with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in ssd_boxes]
            labels = [l.to(device) for l in ssd_lbls]
            difficulties = [d.to(device) for d in ssd_difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

            # ssd metrics: Calculate mAP
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

            # fcn metrics
            fcn_out = torch.nn.functional.log_softmax(fcn_out, dim=1)
            label_pred = fcn_out.max(dim=1)[1].data.cpu().numpy()
            label_true = fcn_lbls.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, fcn_classes)
                fcn_acc_meter.update(acc)
                fcn_acc_cls_meter.update(acc_cls)
                fcn_mean_iu_meter.update(mean_iu)
                fcn_fwavacc_meter.update(fwavacc)

            # print AP for each class
            pp.pprint(APs)

            batch_time.update(time.time() - start)

            start = time.time()

            print(f'BATCH TIME:{batch_time.val:.5f}s  DATA TIME:{data_time.val:.5f}s  {i}th Iteration: Eval Results: '
                  f'Fcn Loss:{fcn_loss_meter.avg:.5f}; Fcn Acc: {fcn_acc_meter.avg:.5f}; Fcn Mean IU:{fcn_mean_iu_meter.avg:.5f}; '
                  f'SSD Loss:{ssd_loss:.5f}; SSD mAP:{mAP:.3f};\n')

        print('Evaluation process has done!')
        del predicted_locs, predicted_scores, fcn_out, _, data

if __name__ == '__main__':

    # valid training results
    eval(valloader=val_loader,
         model = mtlmodel,
         fcn_criterion=fcn_criterion,
         ssd_criterion=ssd_criterion,
         )
