import torch
from model.MtlModel import MtlModel
from model.MultiBoxLoss import MultiBoxLoss
from datasets.MtlDataset import MtlDataset
from torch.utils.data import DataLoader
from utils import *
import time
import torchvision
from metrics import *
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config
data_folder = '/home/chsy1996/Downloads/datasets/'
batch_size = 6
epochs = 50
lr = 1e-3
decay_lr_at_epoch = [6, 8, 14, 28]
decay_lr_factor = 0.1
momentum = 0.9
grad_clip = None

# if checkpoint is not None, training process will start from checkpoint_mtl.pth.tar file
# checkpoint = './checkpoint_mtl.pth.tar'
checkpoint = None

fcn_classes = 6
ssd_classes = 7




def main():

    global start_epoch, label_map, epoch, checkpoint

    if checkpoint is None:
        start_epoch = 0
        mtlmodel = MtlModel(fcn_classes=fcn_classes, ssd_classes=ssd_classes)
        optimizer = torch.optim.SGD(mtlmodel.parameters(),lr=lr,momentum=momentum)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
        mtlmodel = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # move to default device
    mtlmodel = mtlmodel.to(device)
    fcn_criterion = torch.nn.CrossEntropyLoss().to(device)
    ssd_criterion = MultiBoxLoss(priors_cxcy=mtlmodel.priors_cxcy).to(device)

    train_data = MtlDataset(data_folder=data_folder, split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn, shuffle=True)

    val_data = MtlDataset(data_folder=data_folder, split='val')
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=val_data.collate_fn, shuffle=True)

    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at_epoch:
            adjust_learning_rate(optimizer, decay_lr_factor)

        train(trainloader=train_loader,
              model=mtlmodel,
              fcn_criterion=fcn_criterion,
              ssd_criterion=ssd_criterion,
              optimizer=optimizer,
              epoch=epoch
              )

        # Save checkpoint
        save_checkpoint(epoch=epoch,
                        model=mtlmodel,
                        optimizer=optimizer
                        )

        eval(valloader=val_loader,
             eval_time = 10,
             checkpoint='./checkpoint_mtl.pth.tar',
             fcn_criterion=fcn_criterion,
             ssd_criterion=ssd_criterion,
             epoch=epoch
             )

def train(trainloader, model, fcn_criterion,
            ssd_criterion,optimizer, epoch):
    """
    One epoch's training.

    :param trainloader:
    :param valloader:
    :param model:
    :param fcn_criterion:
    :param ssd_criterion:
    :param fcn_optimizer:
    :param ssd_optimizer:
    :param epoch:
    :return:
    """


    model = model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    fcn_losses = AverageMeter()
    ssd_losses = AverageMeter()

    start = time.time()
    # Batches
    for i, data in enumerate(trainloader):

        data_time.update(time.time() - start)

        # fcn

        # get data and label
        fcn_imgs = data[0].to(device) # (B, N, H, W) # float32
        fcn_lbls = data[1].to(device) # (B, H, W) # int64

        # forward propagation
        fcn_out, _, _= model(fcn_imgs)

        # get loss
        loss = fcn_criterion(fcn_out, fcn_lbls)
        fcn_loss = loss
        # clean grad
        optimizer.zero_grad()

        # fcn backward propagation
        loss.backward()

        # SSD

        # get data
        ssd_imgs = data[2].to(device)
        ssd_boxes = [b.to(device) for b in data[3]]
        ssd_labels = [l.to(device) for l in data[4]]

        # forward propa
        _, predicted_locs, predicted_scores = model(ssd_imgs) # tensor size (B, 8732, 4)  tensor size (B, 8732, classes)

        # get loss
        loss = ssd_criterion(predicted_locs, predicted_scores, ssd_boxes, ssd_labels)

        # backward
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # average grad of base layers
        for m in model.shared_backbone.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.grad *= 0.5
            # if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            #     m.weight.grad *= 0.5
            #     m.bias.grad *= 0.5
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.grad *= 0.5
                m.bias.grad *= 0.5

        # update model
        optimizer.step()

        fcn_losses.update(fcn_loss.item(), fcn_imgs.size(0))
        ssd_losses.update(loss.item(), ssd_imgs.size(0))

        batch_time.update(time.time() - start)

        start = time.time()

        print(f'BATCH TIME:{batch_time.val:.5f}s  DATA TIME:{data_time.val:.5f}s  {i}th ITERATION of Epoch {epoch}: Train Results: '
              f'FCN Loss: {fcn_losses.avg:.5f}; '
              f'SSD Loss: {ssd_losses.avg:.5f};\n')

    del predicted_locs, predicted_scores, fcn_out, _, data


def eval(valloader, eval_time, checkpoint, fcn_criterion, ssd_criterion, epoch):
    """

    :param valloader:
    :param model:
    :param fcn_criterion:
    :param ssd_criterion:
    :param optimizer:
    :param epoch:
    :return:
    """

    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.eval()

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
            if i >= eval_time:
                break

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

            print(f'BATCH TIME:{batch_time.val:.5f}s  DATA TIME:{data_time.val:.5f}s  {i}th Iteration of Epoch{epoch}: Eval Results: '
                  f'Fcn Loss:{fcn_loss_meter.avg:.5f}; Fcn Acc: {fcn_acc_meter.avg:.5f}; Fcn Mean IU:{fcn_mean_iu_meter.avg:.5f}; '
                  f'SSD Loss:{ssd_loss:.5f}; SSD mAP:{mAP:.3f};\n')

        print(f'{eval_time} times eval processes have done!\n')
        del predicted_locs, predicted_scores, fcn_out, _, data

if __name__ == '__main__':
    main()