import glob
import numpy as np
import torch
import pandas as pd
import cv2
import torchvision
import os
import sys
import matplotlib.pyplot as plt
from torch import nn, Tensor
import matplotlib.patches as patches
from evaluation import mean_average_precision
import argparse
from torchvision.models.detection import FasterRCNN
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import convert
from torchvision.models.quantization import mobilenet_v3_large
from torchvision.models.mobilenetv3 import ConvBNActivation
from torchvision.models.quantization.mobilenetv3 import QuantizableSqueezeExcitation
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from typing import List, Any, Optional, Callable, Sequence 
import sys
import math
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch import nn
from torch.hub import load_state_dict_from_url
from functools import partial
import ray
from ray.tune.schedulers import ASHAScheduler
from ray import tune

class QuantizableBackbone(nn.Module):
  def __init__(self, backbone):
    super(QuantizableBackbone, self).__init__()
    self.quant = QuantStub()
    self.dequant = DeQuantStub()
    self.out_channels = 960
    self.backbone = backbone

  def forward(self, x):
    x = self.quant(x)
    x = self.backbone(x)
    x = self.dequant(x)
    return x

  def fuse_model(self):
	  for m in self.modules():
	    if type(m) == ConvBNActivation:
	      modules_to_fuse = ['0', '1']
	      if type(m[2]) == nn.ReLU:
	        modules_to_fuse.append('2')
	      fuse_modules(m, modules_to_fuse, inplace=True)
	    elif type(m) == QuantizableSqueezeExcitation:
	      m.fuse_model()



class ClockDataset(object):
  def __init__(self, csv_data, image_lst, transforms, id_map, label_encoder, resize_width, resize_height):
    self.transforms = transforms
    self.csv_data = csv_data
    self.image_lst = image_lst
    self.id_map = id_map
    self.label_encoder = label_encoder
    self.resize_width = resize_width
    self.resize_height = resize_height

  def __getitem__(self, index):
    im_dct = self.image_lst[index]
    filename = im_dct['filename']
    im = np.copy(im_dct['val'])
    im_id = self.id_map[filename]  # unique id for image
    im = cv2.resize(im, dsize=(self.resize_height, self.resize_width), interpolation=cv2.INTER_NEAREST)
    obj_details = self.csv_data[self.csv_data['filename'] == filename]
    boxes = []
    labels = []

    for _, row in obj_details.iterrows():
      im_width = row['width']
      im_height = row['height']
      width_scale = self.resize_width / im_width
      height_scale = self.resize_height / im_height
      xmin = row['xmin'] * width_scale
      xmax = row['xmax'] * width_scale
      ymin = row['ymin'] * height_scale
      ymax = row['ymax'] * height_scale
      label = self.label_encoder[row['class']]
      boxes.append([xmin, ymin, xmax, ymax])
      labels.append(label)

    # image augment returns transformed image and updated bboxes. Takes in [[list: boxes]], [labels] and image
    # im, obj_details = ImageAugment.transform(im, obj_details). returns image and bbox updated

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    if self.transforms is not None:
      im = self.transforms(im)
    target = {}
    target['boxes'] = boxes
    target['labels'] = labels
    target['image_id'] = torch.tensor([im_id])
    return im , target

  def __len__(self):
    return len(self.image_lst)


# custom collate function
def collate_fn(batch):
  data = [item[0] for item in batch]  # image
  target = [item[1] for item in batch]  # dictionary
  return [data, target]

@torch.no_grad()
def eval(model, dataloader_test, device):
  val_losses = [] # average loss per validation set.
  prediction_boxes = []
  ground_truth_boxes = []
  for _, (images, targets) in enumerate(dataloader_test):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k,v in t.items()} for t in targets] # [[boxes: [N,4], labels: [N], filename: __ ], [boxes: [N,4], labels: [N], filename: __], ...]

    model.train()
    # get validation loss
    val_loss_dict = model(images, targets)
    val_loss = sum(loss for loss in val_loss_dict.values())
    val_loss = val_loss.item()
    val_losses.append(val_loss)

    model.eval()
    predictions = model(images) # predictions = [[boxes: [[], ...], labels: [N,...], scores: [N, ...]], [boxes: [[], ...], labels: [N,...], scores: [...]], .... ]
    for i in range(len(predictions)): # len(predictions) = no. of images per batch
      prediction = predictions[i] # one image
      target = targets[i]
      image_id = target['image_id'].item()
      for j in range(len(prediction['boxes'])):
        single_pred = []
        single_pred.append(image_id)
        single_pred.append(prediction['labels'][j])
        single_pred.append(prediction['scores'][j])
        single_pred.extend(prediction['boxes'][j])
        prediction_boxes.append(single_pred)  #[[im_id, class_pred, prob_score, x1, y1, x2, y2], [....]]
        #save_predicts.append(list.copy(single_pred))

      for j in range(len(target['boxes'])):
        single_gt = []
        single_gt.append(image_id)
        single_gt.append(target['labels'][j])
        single_gt.append(1) # dummy value for prob_score
        single_gt.extend(target['boxes'][j])
        ground_truth_boxes.append(single_gt)

  # coco metric
  mAP = 0
  ious = np.arange(0.8, 0.95, 0.05)
  for iou_threshold in ious:
    AP = mean_average_precision(prediction_boxes, ground_truth_boxes, iou_threshold, num_classes = 22)
    mAP += AP
  mAP = mAP / len(ious)
  return sum(val_losses) / len(val_losses), mAP, prediction_boxes

def save_predictions(prediction_boxes, save_path, im_test_id_name_map):
  prediction_boxes = list(map(lambda x: [im_test_id_name_map[x[0]]] + x[1:], prediction_boxes))
  prediction_boxes = list(map(lambda x: [x[0], x[1].item(), x[2].item(), x[3].item(), x[4].item(), x[5].item(), x[6].item()], prediction_boxes))
  columns = ['image','class_pred','prob_score','x1','y1','x2','y2']
  df = pd.DataFrame(prediction_boxes, columns=columns)
  df.to_csv(save_path, index=False)

def train(model, optimizer, data_loader, train_losses, device, transform):
  model.train()
  for step, (images, targets) in enumerate(data_loader):

    images = list(map(lambda x: transform(x), images))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()

    if not math.isfinite(loss_value):
      print("Loss is {}, stopping training".format(loss_value))
      print(loss_dict)
      sys.exit(1)

    train_losses.append(loss_value)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

  return sum(train_losses) / len(train_losses)

def test_model(model, predictions_path, resize_width=320, resize_height=320):
  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
  ])
  device = "cpu"
  #torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset_test = ClockDataset(test_metadata, test_img_list_, transform, im_test_name_id_map, label_encoder, resize_width, resize_height)
  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn = collate_fn)
  test_loss, mAP, prediction_boxes = eval(model, data_loader_test, device)
  print("Test loss: {:5f}, map: {:5f}".format(test_loss, mAP))
  save_predictions(prediction_boxes, predictions_path, im_test_id_name_map)

def prepare_inputs(args):
  train_metadata = pd.read_csv(args.train_csv_path)
  test_metadata = pd.read_csv(args.test_csv_path)
  label_encoder = {"1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "10":10, "11":11, "12":12, "contour":13, "hand":14, "top_left_arrow":15, "top_right_arrow": 16, "right_arrow": 17, "left_arrow": 18, "up_arrow": 19, "down_arrow": 20, "bottom_right_arrow":21, "bottom_left_arrow": 22}

  # load data
  im_train_name_id_map = {}  # 0: image1, 1: image2
  im_val_name_id_map = {}
  im_test_name_id_map = {}

  im_train_id_name_map = {}
  im_val_id_name_map = {}
  im_test_id_name_map = {}

  train_img_list = glob.glob(args.train_im_path +  "/*.png")
  test_img_list = glob.glob(args.test_im_path + "/*.png")

  val_percentage = 0.1
  val_num = math.floor(0.1*len(train_img_list))
  np.random.shuffle(train_img_list)
  train_img_list, val_img_list = train_img_list[val_num: ], train_img_list[: val_num]
  val_img_names = list(map(lambda x: x.split("/")[-1], val_img_list))
  val_metadata = train_metadata[train_metadata['filename'].isin(val_img_names)]

  train_img_list_ = [{} for i in range(len(train_img_list))]
  val_img_list_ = [{} for i in range(len(val_img_list))]
  test_img_list_ = [{} for i in range(len(test_img_list))]

  for i in range(len(train_img_list)):
    im = cv2.imread(train_img_list[i], cv2.IMREAD_GRAYSCALE)
    # convert to 3 channel grayscale
    im = np.repeat(im[..., np.newaxis], 3, -1)
    filename = train_img_list[i].split("/")[-1]
    train_img_list_[i]['filename'] = filename
    train_img_list_[i]['val'] = im

  for i in range(len(val_img_list)):
    im = cv2.imread(val_img_list[i], cv2.IMREAD_GRAYSCALE)
    # convert to 3 channel grayscale
    im = np.repeat(im[..., np.newaxis], 3, -1)
    filename = val_img_list[i].split("/")[-1]
    val_img_list_[i]['filename'] = filename
    val_img_list_[i]['val'] = im

  for i in range(len(test_img_list)):
    im = cv2.imread(test_img_list[i], cv2.IMREAD_GRAYSCALE)
    # convert to 3 channel grayscale
    im = np.repeat(im[..., np.newaxis], 3, -1)
    filename = test_img_list[i].split("/")[-1]
    test_img_list_[i]['filename'] = filename
    test_img_list_[i]['val'] = im

  # id maps when saving predictions
  for i in range(len(train_img_list)):
    filename = train_img_list[i].split("/")[-1]
    im_train_name_id_map[filename] = i
    im_train_id_name_map[i] = filename

  for i in range(len(val_img_list)):
    filename = val_img_list[i].split("/")[-1]
    im_val_name_id_map[filename] = i
    im_val_id_name_map[i] = filename

  for i in range(len(test_img_list)):
    filename = test_img_list[i].split("/")[-1]
    im_test_name_id_map[filename] = i
    im_test_id_name_map[i] = filename

  return train_metadata, val_metadata, test_metadata, \
        label_encoder, train_img_list_, val_img_list_, test_img_list_, \
        im_train_name_id_map, im_train_id_name_map, im_val_name_id_map, \
        im_val_id_name_map, im_test_name_id_map, im_test_id_name_map

def get_model():
  num_classes = 23
  backbone = mobilenet_v3_large(pretrained=True, quantize=False).features
  quantizableBackbone = QuantizableBackbone(backbone)
  anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
  model = FasterRCNN(quantizableBackbone, num_classes=num_classes, rpn_anchor_generator = anchor_generator)
  return model

def print_size_of_model(model):
  torch.save(model.state_dict(), "temp.p")
  print('Size (MB):', os.path.getsize("temp.p")/1e6)
  os.remove('temp.p')

# Training loop
def train_detector(config, checkpoint_dir=None, params=None):
  # model pretrained on COCO dataset

  train_metadata = params['train_metadata']
  val_metadata = params['val_metadata']
  label_encoder = params['label_encoder']
  train_img_list = params['train_img_list']
  val_img_list = params['val_img_list']
  im_train_name_id_map = params['im_train_name_id_map']
  im_train_id_name_map = params['im_train_id_name_map']
  im_val_name_id_map = params['im_val_name_id_map']
  im_val_id_name_map = params['im_val_id_name_map']
  num_epochs = params['num_epochs']
  resize_width = params['resize_width']
  resize_height = params['resize_height']
  model_save_path = params['model_save_path']
  to_tune = params['tune']
  model = get_model()
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
  # transformation. Internally standardise with mean and std already
  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
  ])

  # load dataset
  dataset = ClockDataset(train_metadata, train_img_list, None, im_train_name_id_map, label_encoder, resize_width, resize_height)
  dataset_val = ClockDataset(val_metadata, val_img_list, transform, im_val_name_id_map, label_encoder, resize_width, resize_height)

  # define dataloader
  data_loader = torch.utils.data.DataLoader(dataset, batch_size = config['batch_size'], shuffle=True, collate_fn = collate_fn)
  data_loader_val =  torch.utils.data.DataLoader(dataset_val, batch_size = config['batch_size'], shuffle=False, collate_fn = collate_fn)

  #print_size_of_model(model)

  # quantization
  model.backbone.fuse_model()
  model.backbone.qconfig = torch.quantization.default_qat_qconfig
  model = torch.quantization.prepare_qat(model, inplace=True)
  model.train()
  #print(model)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  train_losses = []  # average loss over all training steps
  for epoch in range(1, num_epochs + 1):
    avg_train_loss = train(model, optimizer, data_loader, train_losses, device, transform)
    avg_val_loss, mAP, prediction_boxes = eval(model, data_loader_val, device)
    print("epoch: {}; Average Train Loss: {:5f}; Average Val Loss: {:5f}; Mean Average Precision: {:5f}"
      .format(epoch, avg_train_loss, avg_val_loss, mAP))

    scheduler.step()
    
    #if epoch % 10 and epoch != 0:
      #if to_tune:
        #with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
          #path = os.path.join(checkpoint_dir, "checkpoint")
          #torch.save((model.state_dict(), optimizer.state_dict()), path)

    if to_tune:
      tune.report(mAP = mAP.item(), loss = avg_val_loss)

  if to_tune:
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((model.state_dict(), optimizer.state_dict()), path)

  if model_save_path:
    torch.save(model.state_dict(), model_save_path)

  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_im_path", type=str, default="./data/train/train_images")
  parser.add_argument("--train_csv_path", type=str, default="./data/train_labels.csv")
  parser.add_argument("--test_im_path", type=str, default="./data/test/test_images")
  parser.add_argument("--test_csv_path", type=str, default="./data/test_labels.csv")
  parser.add_argument("--save_pred_path", type=str, default="./predictions.csv")
  parser.add_argument("--model_save_path", type=str, help="Not required. Only when you want to save model")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num_epochs", type=int, default=16)
  parser.add_argument("--predictions_path", type=str, default="./predictions.csv")
  parser.add_argument("--resize_width", type=int, default=320)
  parser.add_argument("--resize_height", type=int, default=320)
  parser.add_argument("--lr", type=float, help="custom learning rate. Provide when not in tune mode")
  parser.add_argument("--momentum", type=float, help="custom momentum. Provide when not in tune mode")
  parser.add_argument("--weight_decay", type=float, help="custom weight decay. Provide when not in tune mode")
  parser.add_argument("--batch_size", type=int, help="custom batch size. Provide when not in tune mode")
  parser.add_argument("--tune", action='store_true', default=False)
  parser.add_argument("--mixup", action='store_true', default=False)
  parser.add_argument("--augment_data", action='store_true', default=False)
  parser.add_argument("--num_samples", type=int, default=15)
  parser.add_argument("--grace_period", type=int, default=10)
  args = parser.parse_args()

  params = {}

  train_metadata, val_metadata, test_metadata, \
        label_encoder, train_img_list_, val_img_list_, test_img_list_, \
        im_train_name_id_map, im_train_id_name_map, im_val_name_id_map, \
        im_val_id_name_map, im_test_name_id_map, im_test_id_name_map = prepare_inputs(args)

  params['train_metadata'] = train_metadata
  params['val_metadata'] = val_metadata
  params['label_encoder'] = label_encoder
  params['train_img_list'] = train_img_list_ # turn this into a dict {image1: np.array}
  params['val_img_list'] = val_img_list_
  params['im_train_name_id_map'] = im_train_name_id_map
  params['im_train_id_name_map'] = im_train_id_name_map
  params['im_val_name_id_map'] = im_val_name_id_map
  params['im_val_id_name_map'] = im_val_id_name_map
  params['resize_width'] = args.resize_width
  params['resize_height'] = args.resize_height
  params['num_epochs'] = args.num_epochs
  params['model_save_path'] = args.model_save_path
  params['tune'] = args.tune
    
  if args.tune:
    config = {
      "momentum": tune.uniform(0.1, 0.9),
      "weight_decay": tune.loguniform(1e-4, 1e-1),
      "lr": tune.loguniform(1e-4, 1e-1),
      "batch_size": tune.choice([4, 8])
    }
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "1"
    scheduler = ASHAScheduler(
      max_t = args.num_epochs,
      grace_period = args.grace_period,
    )

    result = tune.run(
      tune.with_parameters(train_detector, params=params),
      config=config,
      metric="mAP",
      mode="max",
      num_samples= args.num_samples,
      scheduler=scheduler,
      log_to_file=True,
      resources_per_trial={"cpu":1,"gpu":1},
      verbose=1,
      raise_on_failed_trial=False
    )

    best_trial = result.get_best_trial(metric="mAP", mode="max")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final val loss: {}".format(best_trial.last_result['loss']))
    print("Best trial mAP: {}".format(best_trial.last_result['mAP'])) # in terms of validation loss

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, _ = torch.load(checkpoint_path)
    # load model state 
    trained_model = get_model()
    trained_model.backbone.fuse_model()
    trained_model.backbone.qconfig = torch.quantization.default_qat_qconfig
    trained_model = torch.quantization.prepare_qat(trained_model, inplace=True)
    trained_model.load_state_dict(model_state)

  else: 
    config = {
      "momentum": args.momentum,
      "weight_decay": args.weight_decay,
      "lr": args.lr,
      "batch_size": args.batch_size
    }
    trained_model = train_detector(config,params=params)

  trained_model.cpu()
  trained_model = convert(trained_model, inplace=True)

  trained_model.eval()
  test_model(trained_model, args.predictions_path, resize_width=args.resize_width, resize_height = args.resize_height)
  print_size_of_model(trained_model)

  # convert to torchscript
  scripted_model = torch.jit.script(trained_model)
  torch.jit.save(scripted_model, "mobilenetv3_script.pt")

  optimized_model = optimize_for_mobile(scripted_model)
  optimized_model.save("mobilenetv3_optimized.pt")
