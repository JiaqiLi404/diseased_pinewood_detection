import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import numpy as np

import cv2

img = read_image('misc/demo.jpg')
img = t.from_numpy(img)[None]

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

# in this machine the cupy isn't install correctly...
# so it's a little slow
trainer.load('./chainer_best_model_converted_to_pytorch_0.7053.pth')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))
img=np.transpose(img[0].numpy(),(1,2,0))
img=img.astype(np.uint8)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("h",img)
cv2.waitKey(0)