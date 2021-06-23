import glob
import numpy as np
import torch
import pandas as pd
import cv2
import torchvision
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from evaluation import mean_average_precision
import argparse
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
import math
# classes to detect: Digits 1 - 12
# contour, hand of clock. 13, 14
# [{k:v for k,v in t.items()} for t in targets]. t = {'boxes': [list of list], 'labels': [list of id], 'image_id': [list of id]}


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

def mixup_data(images, targets, device, mixup_ratio=0.5):
  images = torch.stack(images)
  batch_size = images.size()[0]
  index = torch.randperm(batch_size, dtype=int)
  index = index.to(device)
  mixed_images = mixup_ratio * images + (1 - mixup_ratio) * images[index,:]
  targets = np.array(targets)
  index_list = index.tolist()
  targets_ = targets[index_list]
  
  for target, pair_target in zip(targets, targets_):
    target['boxes'] = torch.cat((target['boxes'], pair_target['boxes']))
    target['labels'] = torch.cat((target['labels'], pair_target['labels']))
  return mixed_images, targets


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
  ious = np.arange(0.5, 0.95, 0.05)
  for iou_threshold in ious:
    AP = mean_average_precision(prediction_boxes, ground_truth_boxes, iou_threshold, num_classes = 14)
    mAP += AP
  mAP = mAP / 10
  return sum(val_losses) / len(val_losses), mAP, prediction_boxes

def save_predictions(prediction_boxes, save_path, im_test_id_name_map):
  prediction_boxes = list(map(lambda x: [im_test_id_name_map[x[0]]] + x[1:], prediction_boxes))
  prediction_boxes = list(map(lambda x: [x[0], x[1].item(), x[2].item(), x[3].item(), x[4].item(), x[5].item(), x[6].item()], prediction_boxes))
  columns = ['image','class_pred','prob_score','x1','y1','x2','y2']
  df = pd.DataFrame(prediction_boxes, columns=columns)
  df.to_csv(save_path, index=False)

def train(model, optimizer, data_loader, train_losses, device, mixup):
  model.train()
  for step, (images, targets) in enumerate(data_loader):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

    if mixup:
      images, targets = mixup_data(images, targets, device)
      
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    if mixup:
      losses = 0.5 * losses # mixup ratio
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

def test_model(predictions_path, retina_net, best_trial=None, tune=False, model_save_path=None, resize_width=800, resize_height=800):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
  ])

  best_trained_model = get_model(retina_net)
  # load weights from checkpoint

  if tune:
    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint_path)
  else:
    assert (model_save_path != None)
    model_state = torch.load(model_save_path)  

  best_trained_model.load_state_dict(model_state)
  best_trained_model.to(device)
  dataset_test = ClockDataset(test_metadata, test_img_list_, transform, im_test_name_id_map, label_encoder, resize_width, resize_height)
  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn = collate_fn)
  test_loss, mAP, prediction_boxes = eval(best_trained_model, data_loader_test, device)
  print("Test loss: {:5f}, map: {:5f}".format(test_loss, mAP))
  save_predictions(prediction_boxes, predictions_path, im_test_id_name_map)
  return best_trained_model

def prepare_inputs(args):
  train_metadata = pd.read_csv(args.train_csv_path)
  test_metadata = pd.read_csv(args.test_csv_path)
  label_encoder = {"1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "10":10, "11":11, "12":12, "contour":13, "hand":14, "top_left_arrow":15, "top_right_arrow": 16, "right_arrow": 17, "left_arrow": 18, "up_arrow": 19, "down_arrow": 20, "bottom_right_arrow":21}

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

def get_model(retina_net): 
  num_classes = 22  # 21 + background
  if retina_net: 
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    # replace classification layer 
    out_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors  
    model.head.classification_head.num_classes = num_classes
    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits
  else:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
    # input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace pretrained head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

def save_checkpoint(model, optimizer, tune, checkpoint_dir):
  if tune: 
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save(
          (model.state_dict(), optimizer.state_dict()), path)

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
  mixup = params['mixup']
  tune = params['tune']
  num_epochs = params['num_epochs']
  resize_width = params['resize_width']
  resize_height = params['resize_height']
  model_save_path = params['model_save_path']
  model = get_model(params['retina_net'])

  params = [p for p in model.parameters() if p.requires_grad]

  optimizer = torch.optim.SGD(params, lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs) 
  # transformation. Internally standardise with mean and std already
  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
  ])

  # load dataset
  dataset = ClockDataset(train_metadata, train_img_list, transform, im_train_name_id_map, label_encoder, resize_width, resize_height)
  dataset_val = ClockDataset(val_metadata, val_img_list, transform, im_val_name_id_map, label_encoder, resize_width, resize_height)

  # define dataloader
  data_loader = torch.utils.data.DataLoader(dataset, batch_size = config['batch_size'], shuffle=True, collate_fn = collate_fn)
  data_loader_val =  torch.utils.data.DataLoader(dataset_val, batch_size = config['batch_size'], shuffle=False, collate_fn = collate_fn)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  train_losses = []  # average loss over all training steps
  for epoch in range(1, num_epochs + 1):
    avg_train_loss = train(model, optimizer, data_loader, train_losses, device, mixup)
    avg_val_loss, mAP, prediction_boxes = eval(model, data_loader_val, device)
    print("epoch: {}; Average Train Loss: {:5f}; Average Val Loss: {:5f}; Mean Average Precision: {:5f}"
      .format(epoch, avg_train_loss, avg_val_loss, mAP))

    scheduler.step()
    if epoch % 10 == 0 and epoch != 0:
      save_checkpoint(model, optimizer, tune, checkpoint_dir)

    if tune:
      tune.report(mAP = mAP.item(), loss = avg_val_loss)

  if model_save_path and not args.tune:  
    torch.save(model.state_dict(), model_save_path)

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
  parser.add_argument("--num_samples", type=int, default=15)
  parser.add_argument("--grace_period", type=int, default=3)
  parser.add_argument("--reduction_factor", type=int, default=2)
  parser.add_argument("--predictions_path", type=str, default="./predictions.csv")
  parser.add_argument("--retina_net", action='store_true',  default=False)
  parser.add_argument("--resize_width", type=int, default=800)
  parser.add_argument("--resize_height", type=int, default=800)
  parser.add_argument("--mixup", action='store_true', default=False)
  parser.add_argument("--tune", action='store_true', default=False, help="whether the script should run in tuning mode")
  parser.add_argument("--lr", type=float, help="custom learning rate. Provide when not in tune mode")
  parser.add_argument("--momentum", type=float, help="custom momentum. Provide when not in tune mode")
  parser.add_argument("--weight_decay", type=float, help="custom weight decay. Provide when not in tune mode")
  parser.add_argument("--batch_size", type=int, help="custom batch size. Provide when not in tune mode")
  args = parser.parse_args()


  params = {}

  train_metadata, val_metadata, test_metadata, \
        label_encoder, train_img_list_, val_img_list_, test_img_list_, \
        im_train_name_id_map, im_train_id_name_map, im_val_name_id_map, \
        im_val_id_name_map, im_test_name_id_map, im_test_id_name_map = prepare_inputs(args)

  if args.retina_net:
    print("Using RetinaNet model for experiments")

  params['train_metadata'] = train_metadata
  params['val_metadata'] = val_metadata
  params['label_encoder'] = label_encoder
  params['train_img_list'] = train_img_list_ # turn this into a dict {image1: np.array}
  params['val_img_list'] = val_img_list_
  params['im_train_name_id_map'] = im_train_name_id_map
  params['im_train_id_name_map'] = im_train_id_name_map
  params['im_val_name_id_map'] = im_val_name_id_map
  params['im_val_id_name_map'] = im_val_id_name_map
  params['retina_net'] = args.retina_net
  params['resize_width'] = args.resize_width
  params['resize_height'] = args.resize_height
  params['mixup'] = args.mixup
  params['num_epochs'] = args.num_epochs
  params['tune'] = args.tune
  params['model_save_path'] = args.model_save_path
  if args.tune:
    config = {
          "momentum": tune.uniform(0.1, 0.9),
          "weight_decay": tune.loguniform(1e-4, 1e-1),
          "lr": tune.loguniform(1e-4, 1e-1),
          "batch_size": tune.choice([2, 4, 8])
    }
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "1"
    scheduler = ASHAScheduler(
      max_t = args.max_num_epochs,
      grace_period= args.grace_period,
      reduction_factor= args.reduction_factor,
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
    print("Best trial mAP: {}".format(best_trial.last_result['mAP']))
    test_model(args.predictions_path, args.retina_net, tune=args.tune, best_trial=best_trial)
    model.save(model.state_dict(), args.model_save_path)

  else:
    config = {
      "momentum": args.momentum,
      "weight_decay": args.weight_decay,
      "lr": args.lr,
      "batch_size": args.batch_size
    }
    train_detector(config,params=params)
    test_model(args.predictions_path, args.retina_net, tune=args.tune, model_save_path=args.model_save_path)
