import torchvision.transforms.functional as FT
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torchvision import transforms as tfs
from datasets.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from checkpoint
checkpoint = './checkpoint_mtl.pth.tar'
checkpoint = torch.load(checkpoint)
start_point = checkpoint['epoch'] + 1
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = tfs.Resize((300, 300))
to_tensor = tfs.ToTensor()
normalize = tfs.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

print(f'{start_point} epoches model results:')

def _predict(image_path, min_score=0.2, max_overlap=0.5, top_k=200, suppress=None): # predict result

    # segmentation results:

    # process image to tensor so that we can feed it into model
    RGB_image = Image.open(image_path)
    image = normalize(to_tensor(resize(RGB_image))).to(device)

    # forward propa.
    seg_out, predicted_locs, predicted_scores = model(image.unsqueeze(dim=0))    # (1, n_classes, 300, 300)  (1, 8732, 4)    (1, 8732, n_classes)

    # seg pred
    seg_pred = seg_out.max(dim=1)[1].squeeze().cpu().data.numpy()   # (300, 300)    int64
    dd_cm = np.array(dd_colormap, dtype='uint8')
    seg_pred = dd_cm[seg_pred]  # RGB color numpy
    seg_pred_img = Image.fromarray(seg_pred)


    # Detection pred:
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original dimensions
    original_dims = torch.FloatTensor([RGB_image.width, RGB_image.height, RGB_image.width, RGB_image.height]).unsqueeze(0)

    det_boxes = det_boxes * original_dims   # (4, n_boxes)

    # Decode class integer labels
    det_labels = [rev_sdd_labelmap[l] for l in det_labels[0].to('cpu').tolist()]    # list of labels

    # If no objects found, the detected labels will be set to ['0.']
    if det_labels == ['background']:
        return seg_pred_img, RGB_image

    # Annotate
    annotate_image = RGB_image
    draw = ImageDraw.Draw(annotate_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):  # number of predicted boxes
        if suppress is not None:
            if det_labels[i] in suppress:
                continue


        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=sdd_label_colormap[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=sdd_label_colormap[det_labels[i]])
        # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 1. for l in box_location], outline=sdd_label_colormap[det_labels[i]])
        # draw.rectangle(xy=[l + 1. for l in box_location], outline=sdd_label_colormap[det_labels[i]])

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=sdd_label_colormap[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    del draw

    return seg_pred_img, annotate_image

def _show_seg_label(data_path):

    lbl = Image.open(data_path)
    lbl_category = FT.to_tensor(lbl) * 255

    if lbl_category.shape[0] == 3:
        lbl_category = lbl_category[0, :, :]  # (300, 300)
    lbl_category = lbl_category.data.numpy()
    lbl_category = lbl_category.astype('int')  # int64

    dd_cm = np.array(dd_colormap, dtype='uint8')

    lbl_RGB = dd_cm[lbl_category]

    lbl_RGB = Image.fromarray(lbl_RGB)
    Image._show(lbl_RGB)

if __name__ == '__main__':
    # Inference Image path
    # img_path = '/home/chsy1996/Downloads/datasets/dataset-medium/image-chips/1d056881e8_29FEA32BC7INSPIRE-000128.png'
    img_path = '/home/chsy1996/Downloads/datasets/StanfordDroneDataset/sdd/JPEGImages/deathCircle_video0_5.jpg'

    checkpoint = './checkpoint_mtl.pth.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']

    seg_pred_img, det_pred_img = _predict(image_path=img_path)
    Image._show(seg_pred_img)
    Image._show(det_pred_img)

    # if Image has correspond label, show it
    # seg_lbl_path = '/home/chsy1996/Downloads/datasets/dataset-medium/label-chips/1d056881e8_29FEA32BC7INSPIRE-000128.png'
    # _show_seg_label(seg_lbl_path)
