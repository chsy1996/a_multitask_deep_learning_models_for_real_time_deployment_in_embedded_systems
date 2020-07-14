# In source code, the author add one during Inference, we directly remove Ignore class here because we totally needn't Ignore class,
# you can find 'add one' operator in Inference class in This link: https://github.com/dronedeploy/dd-ml-segmentation-benchmark/blob/master/libs/Inference.py
# DroneDeploy dataset
dd_classes = ['Building', 'Clutter', 'Vegetation', 'Water', 'Ground', 'Car']
dd_colormap = [[230, 25, 75], [145, 30, 180], [60, 180, 75], [245, 130, 48], [255, 255, 255], [0, 130, 200]]


# StanfordDroneDataset
sdd_labels = ('biker', 'pedestrian', 'skater', 'cart', 'car', 'bus')
sdd_labelmap = {k: v + 1 for v, k in enumerate(sdd_labels)}
sdd_labelmap['background'] = 0
rev_sdd_labelmap = {v: k for k,v in sdd_labelmap.items()}

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#f58231', '#f032e6', '#aa6e28', '#808080']

sdd_label_colormap = {k: distinct_colors[i] for i,k in enumerate(sdd_labelmap.keys())}
