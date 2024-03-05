import os
import numpy as np
import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as TT
import wget
import cv2
import random
import warnings
import transforms as T
import utils

from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate

warnings.filterwarnings('ignore')

class CocoDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_dir, subset, transforms):
      dataset_path = os.path.join(dataset_dir, subset)
      ann_file = os.path.join(dataset_path, "annotation.json")
      self.imgs_dir = os.path.join(dataset_path, "images")
      self.coco = COCO(ann_file)
      self.img_ids = self.coco.getImgIds()
      self.transforms = transforms


  def __getitem__(self, idx):
      '''
      Args:
          idx: index of sample to be fed
      return:
          dict containing:
          - PIL Image of shape (H, W)
          - target (dict) containing:
              - boxes:    FloatTensor[N, 4], N being the n° of instances and it's bounding
                          boxes coordinates in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H;
              - labels:   Int64Tensor[N], class label (0 is background);
              - image_id: Int64Tensor[1], unique id for each image;
              - area:     Tensor[N], area of bbox;
              - iscrowd:  UInt8Tensor[N], True or False;
              - masks:    UInt8Tensor[N, H, W], segmantation maps;
      '''
      img_id = self.img_ids[idx]
      img_obj = self.coco.loadImgs(img_id)[0]
      anns_obj = self.coco.loadAnns(self.coco.getAnnIds(img_id))

      img = Image.open(os.path.join(self.imgs_dir, img_obj['file_name']))

      # list comprhenssion is too slow, might be better changing it
      #bboxes = [ann['bbox'] for ann in anns_obj]
      masks = [self.coco.annToMask(ann) for ann in anns_obj]
      areas = [ann['area'] for ann in anns_obj]

      num_objs = len(anns_obj)
      boxes = []
      for i in range(num_objs):
          xmin = anns_obj[i]['bbox'][0]
          ymin = anns_obj[i]['bbox'][1]
          xmax = xmin + anns_obj[i]['bbox'][2]
          ymax = ymin + anns_obj[i]['bbox'][3]
          boxes.append([xmin, ymin, xmax, ymax])

      boxes = torch.as_tensor(boxes, dtype=torch.float32)
      labels = torch.ones(num_objs, dtype=torch.int64)
      masks = torch.as_tensor(masks, dtype=torch.uint8)
      image_id = torch.tensor([idx])
      area = torch.as_tensor(areas)
      iscrowd = torch.zeros(num_objs, dtype=torch.int64)

      target = {}
      target["boxes"] = boxes
      target["labels"] = labels
      target["masks"] = masks
      target["image_id"] = image_id
      target["area"] = area
      target["iscrowd"] = iscrowd

      if self.transforms is not None:
          img, target = self.transforms(img, target)
      return img, target


  def __len__(self):
      return len(self.img_ids)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    #transforms.append(T.ToTensor())
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img_path, model, device, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    CLASS_NAMES = ['__background__', 'packet']
    img = Image.open(img_path)
    transform = TT.Compose([TT.ToTensor()])
    img = transform(img)

    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy().astype(int))]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def segment_instance(img_path, model, device, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, model, device, confidence)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
      rgb_mask = get_coloured_mask(masks[i])
      img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(10,15))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    #torch.cuda.empty_cache()

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = CocoDataset('./','dataset', get_transform(train=True))
    dataset_test = CocoDataset('./','dataset', get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-10])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    
    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("Training done")

    #torch.save(model.state_dict(), './mask-rcnn-refill.pt')

    # set to evaluation mode
    model.eval()

    #response = wget.download("https://media.karousell.com/media/photos/products/2023/5/17/pigeon_liquid_cleanser_refill__1684303535_5e959cea_progressive.jpg", "test.jpg")
    #response = wget.download("https://d1gvm6reez0dkh.cloudfront.net/40047837347/35849996-1644983475.9104.jpg", "test.jpg")
    segment_instance('test.jpg', model, device, confidence=0.5)

if __name__ == "__main__":
    main()
