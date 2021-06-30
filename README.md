# clock_scoring

This project aims to automate the scoring of hand drawn clocks for participants who take part in the Montreal Cognitive Assessment (MoCA) or Mini-Mental State Examination (MMSE) test. 

**Examples of hand drawn clocks by participants:**

<img src="data_v1.1/image_data/27032021145144_NCI_2_1_1_1.png" width="200" height="200"/> <img src="data_v1.1/image_data/27032021140152_MiD_1_1_1_0.png" width="200" height="200"/> <img src="data_v1.1/image_data/27032021145144_NCI_2_1_1_1.png" width="200" height="200"/> <img src="data_v1.1/image_data/27032021154411_MCI_1_1_0_0.png" width="200" height="200"/> <img src="data_v1.1/image_data/HFIXGJBW_NCI_1_1_1_1.png" width="200" height="200"/> <img src="data_v1.1/image_data/IOWLFOXW_NCI_2_1_1_0.png" width="200" height="200"/> <img src="data_v1.1/image_data/PFPVMFKG_MiD_1_1_1_1.png" width="200" height="200"/> <img src="data_v1.1/image_data/YPLLPJVQ_MCI_2_1_1_0.png" width="200" height="200"/> 

## Scoring criteria
**Scoring:** One point is allocated for each of the following three criteria:
* Contour (1 pt): The clock face must be a circle with only minor distortion acceptable (e.g slight imperfection on closing the circle);
* Numbers (1 pt): All clock numbers must be present with no additional numbers; numbers must be in the correct order and placed in the approximate quadrants on the clock face; Roman numerals are acceptable; numbers can be placed outside the circle contour
* Hands (1 pt): There must be two hands jointly indicating the correct time; the hour hand must be clearly shorter than the minute hand; hands must be centered within the clock face with their junction clsoe to the clock centre

## Methodology
We integrate an object detection model into the scoring pipeline. Object detection is suited for such a task because of the scoring of the clock is also based upon the existence of clock elements (Eg. Digits, clock hand and clock face). The bounding boxes of each element from the detection model also aids in the localisation of elements, which in turn streamlines the process of developing a scoring algorithm. 

### Experiments
We perform an ablation study on various object detection models available in pytorch; Namely, the faster - rcnn and Retina Net model. We use the mean average precision (mAP) over all IOU thresholds from 0.5:0.95 as the metric to optimise for. (which is one of the metrics used in the COCO dataset object detection challenge). We initialise the model with weights pretrained from COCO dataset and fine tune on our own clock drawing dataset.
