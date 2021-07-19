# clock_scoring

## Experimental results
All object detection models use the ResNet-50 backbone with feature pyramid network (FPN). Experimental results run for 5 times and the mAP at IOU(0.8:0.9:0.05) is averaged. 

Training parameters
`lr = 0.01`, `weight_decay = 0.0005`, `momentum = 0.9`, `batch_size = 8`, `num_epochs = 20`


| Model      | mAP (%) |      SE of mean     |
| ----------- | ----------- | ---------- |
| Faster-RCNN   | 69.94     |    0.348   |
| Faster-RCNN w mixup     | 70.45      |     0.288       |
| Faster-RCNN w aug  | 69.03   |    0.467   |
| Faster-RCNN w mixup     | 70.45      |     0.288       |
| RetinaNet   | 71.29     |    0.382   |
| RetinaNet w mixup     | 68.10      |     0.109       |
| RetinaNet w aug     | 69.94      |     0.401       |
