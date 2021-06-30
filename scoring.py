import pandas as pd
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from enum import Enum
from PIL import Image, ImageDraw
import glob
import os
OBJ_THRESH = 0.6 # threshold to determine bbox has object
HAND_X_WIDTH = 150 # threshold on how far away mid point of hand must be from contour center before not considered middle
HAND_Y_HEIGHT = 150
LONG_SHORT_THRESH = 5

class Classes(Enum):
	ONE = 1
	TWO = 2
	THREE = 3
	FOUR = 4
	FIVE = 5
	SIX = 6
	SEVEN = 7
	EIGHT = 8
	NINE = 9
	TEN = 10
	ELEVEN = 11
	TWELEVE = 12
	CONTOUR = 13
	HAND = 14
	TOP_LEFT_ARROW = 15
	TOP_RIGHT_ARROW = 16
	RIGHT_ARROW = 17
	LEFT_ARROW = 18
	UP_ARROW = 19
	DOWN_ARROW = 20
	BOTTOM_RIGHT_ARROW = 21
	BOTTOM_LEFT_ARROW = 22


def convert(im_pred):
	class_dct = {}
	for _, row in im_pred.iterrows():
		class_pred = Classes(row['class_pred'])
		x1 = row['x1']
		x2 = row['x2']
		y1 = row['y1']
		y2 = row['y2']
		bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
		class_dct[class_pred] = bbox
	return class_dct
# time is in dict: {"hour": 11, "minute": 10}. Accept multiple of 5

# returns the head and tail coordinates of the sub hands
def get_arrow_coords(arrow_obj, arrow_bbox):
	if arrow_obj == Classes.TOP_LEFT_ARROW:
		top_left_x = arrow_bbox['x1']
		top_left_y = arrow_bbox['y1']
		tail_x = arrow_bbox['x2']
		tail_y = arrow_bbox['y2']
		return top_left_x, top_left_y, tail_x, tail_y
	elif arrow_obj == Classes.TOP_RIGHT_ARROW:
		top_right_x = arrow_bbox['x2']
		top_right_y = arrow_bbox['y1']
		tail_x = arrow_bbox['x1']
		tail_y = arrow_bbox['y2']
		return top_right_x, top_right_y, tail_x, tail_y
	elif arrow_obj == Classes.RIGHT_ARROW:
		top_right_x = arrow_bbox['x2']
		top_right_y = arrow_bbox['y1']
		bot_right_x = arrow_bbox['x2']
		bot_right_y = arrow_bbox['y2']
		tail_x = arrow_bbox['x1']
		return top_right_x, (top_right_y + bot_right_y) // 2, tail_x, (top_right_y + bot_right_y) // 2
	elif arrow_obj == Classes.LEFT_ARROW:
		top_left_x = arrow_bbox['x1']
		top_left_y = arrow_bbox['y1']
		bot_left_x = arrow_bbox['x1']
		bot_left_y = arrow_bbox['y2']
		tail_x = arrow_bbox['x2']
		return top_left_x, (top_left_y + bot_left_y) // 2, tail_x, (top_left_y + bot_left_y) // 2
	elif arrow_obj == Classes.UP_ARROW:
		top_left_x = arrow_bbox['x1']
		top_left_y = arrow_bbox['y1']
		top_right_x = arrow_bbox['x2']
		top_right_y = arrow_bbox['y1']
		tail_y = arrow_bbox['y2']
		return (top_left_x + top_right_x) // 2, top_left_y, (top_left_x + top_right_x) // 2, tail_y
	elif arrow_obj == Classes.DOWN_ARROW:
		bot_left_x = arrow_bbox['x1']
		bot_left_y = arrow_bbox['y2']
		bot_right_x = arrow_bbox['x2']
		bot_right_y = arrow_bbox['y2']
		tail_y = arrow_bbox['y1']
		return (bot_left_x + bot_right_x) // 2, bot_left_y, (bot_left_x + bot_right_x) // 2, tail_y
	elif arrow_obj == Classes.BOTTOM_RIGHT_ARROW:
		bot_right_x = arrow_bbox['x2']
		bot_right_y = arrow_bbox['y2']
		tail_x = arrow_bbox['x1']
		tail_y = arrow_bbox['y1']
		return bot_right_x, bot_right_y, tail_x, tail_y
	elif arrow_obj == Classes.BOTTOM_LEFT_ARROW:
		bot_left_x = arrow_bbox['x1']
		bot_left_y = arrow_bbox['y2']
		tail_x = arrow_bbox['x2']
		tail_y = arrow_bbox['y1']
		return bot_left_x, bot_left_y, tail_x, tail_y

def get_digit_center_pos(class_dct):
	# check whether hands are pointing to correct time
	digits = [digit for digit in range(1, 13)]
	digits_center_pos = {}
	for key in class_dct:
		if key.value in digits:
			digit_bbox = class_dct[key]
			digit_x1, digit_y1, digit_x2, digit_y2 = digit_bbox['x1'], digit_bbox['y1'], digit_bbox['x2'], digit_bbox['y2']
			digit_center_x, digit_center_y = (digit_x1 + digit_x2) // 2, (digit_y1 + digit_y2) // 2
			digits_center_pos[key] = (digit_center_x, digit_center_y)
	return digits_center_pos

# class_dct : {enum : bbox}
def score_hand(class_dct, image, time, im_draw=False):
	draw = ImageDraw.Draw(image)
	msg = ""
	hour = time['hour']
	minute = time['minute']
	if minute == 0:
		minute = 12
	else:
		minute = minute // 5

	one_clock_hand = list(class_dct.keys()).count(Classes.HAND) == 1
	correct_sub_hands = sum(i.value > 14 for i in class_dct.keys()) == 2

	if not one_clock_hand or not correct_sub_hands:
		msg = "did not detect clock hand(14) or two sub hands"
		return 0, msg

	contour_bbox = class_dct[Classes.CONTOUR]
	contour_mid_x, contour_mid_y = int((contour_bbox['x1'] + contour_bbox['x2']) // 2), int((contour_bbox['y1'] + contour_bbox['y2']) // 2)

	# check whether long and short hand distinguishable
	sub_hands = [key for key in class_dct.keys() if key.value > 14]
	#print(sub_hands)
	sub_hand1_cls, sub_hand2_cls = sub_hands[0], sub_hands[1]
	sub_hand1_bbox, sub_hand2_bbox = class_dct[sub_hand1_cls], class_dct[sub_hand2_cls]
	sub_hand1_dist = math.hypot(sub_hand1_bbox['x1'] - sub_hand1_bbox['x2'], sub_hand1_bbox['y1'] - sub_hand1_bbox['y2'])
	sub_hand2_dist = math.hypot(sub_hand2_bbox['x1'] - sub_hand2_bbox['x2'], sub_hand2_bbox['y1'] - sub_hand2_bbox['y2'])
	if abs(sub_hand1_dist - sub_hand2_dist) <= LONG_SHORT_THRESH:
		msg = "hands are not distinguishable"
		return msg, 0

	sub_hand_dct = {}
	if sub_hand1_dist > sub_hand2_dist:
		sub_hand_dct['long'] = (sub_hand1_cls, sub_hand1_bbox)
		sub_hand_dct['short'] = (sub_hand2_cls, sub_hand2_bbox)
	else:
		sub_hand_dct['long'] = (sub_hand2_cls, sub_hand2_bbox)
		sub_hand_dct['short'] = (sub_hand1_cls, sub_hand1_bbox)

	digits_center_pos = get_digit_center_pos(class_dct)	# return dictionary of digits as key and center coords as value

	long_hand_cls, long_hand_bbox = sub_hand_dct['long']
	short_hand_cls, short_hand_bbox = sub_hand_dct['short']

	# which corner to take depends on the class of the arrow
	long_arrow_cls, short_arrow_cls = Classes(long_hand_cls), Classes(short_hand_cls)

	long_head_x, long_head_y, long_tail_x, long_tail_y = get_arrow_coords(long_arrow_cls, long_hand_bbox)
	short_head_x, short_head_y, short_tail_x, short_tail_y = get_arrow_coords(short_arrow_cls, short_hand_bbox)

	# compute distances of arrow head coords to digits
	long_digit_dist = {key: math.hypot(long_head_x - digits_center_pos[key][0],
		long_head_y - digits_center_pos[key][1]) for key in digits_center_pos}
	short_digit_dist = {key: math.hypot(short_head_x - digits_center_pos[key][0],
		short_head_y - digits_center_pos[key][1]) for key in digits_center_pos}

	tail_mid_x, tail_mid_y = (long_tail_x + short_tail_x) // 2, (long_tail_y + short_tail_y) // 2

	if im_draw:
		draw.point([(tail_mid_x, tail_mid_y)], fill='blue')
		draw.point([(contour_mid_x, contour_mid_y)], fill='red')
		draw.rectangle([contour_mid_x - (HAND_X_WIDTH // 2), contour_mid_y - (HAND_Y_HEIGHT // 2),
			contour_mid_x + (HAND_X_WIDTH // 2), contour_mid_y + (HAND_Y_HEIGHT // 2)], outline='red')
	nearest_digit_long_arrow, nearest_digit_short_arrow = min(long_digit_dist, key=long_digit_dist.get), \
		min(short_digit_dist, key=short_digit_dist.get)
	#print(nearest_digit_long_arrow)
	#print(nearest_digit_short_arrow)
	if (not contour_mid_x - (HAND_X_WIDTH // 2) < tail_mid_x < contour_mid_x + (HAND_X_WIDTH // 2) or \
			not contour_mid_y - (HAND_Y_HEIGHT // 2) < tail_mid_y < contour_mid_y + (HAND_Y_HEIGHT // 2)):
		return 0, "hand not centered"

	if im_draw:
		bbox_long_arrow, bbox_short_arrow = class_dct[long_arrow_cls], class_dct[short_arrow_cls]
		draw.rectangle(list(bbox_long_arrow.values()), outline='red')
		draw.rectangle(list(bbox_short_arrow.values()), outline='red')
		draw.line([digits_center_pos[nearest_digit_long_arrow],
			(long_head_x, long_head_y)], fill='green', width=2)
		draw.line([digits_center_pos[nearest_digit_short_arrow],
			(short_head_x, short_head_y)], fill='green', width=2)

	if nearest_digit_long_arrow.value == minute and nearest_digit_short_arrow.value == hour:
		return 1, "passed"
	else:
		return 0, "long hand or short hand not pointing to correct digit"

def compute_angle(vec1, vec2):
	dot_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
	vec1_mag = math.hypot(vec1[0], vec1[1])
	vec2_mag = math.hypot(vec2[0], vec2[1])
	cos_angle = dot_prod / (vec1_mag * vec2_mag)
	angle = math.acos(cos_angle)
	return angle * (180/math.pi) # in radians

def score_digits(class_dct, image, im_draw = False):
	# compute angles for each digits. Arrange the angles in ascending order. Angles should be in order of digit
	msg = ""
	digits = [digit for digit in class_dct.keys() if digit.value < 13]
	if len(set(digits)) != 12:	# exactly 12 digits
		msg += "did not detect all 12 digits"
		return 0, msg
	digits_center_pos = get_digit_center_pos(class_dct)

	highest_digit_in_y = min(digits_center_pos, key = lambda x: digits_center_pos[x][1])
	if highest_digit_in_y != Classes.TWELEVE:
		msg = "Digit 12 not in correct order"
		return 0, msg
	#print(highest_digit_in_y)

	contour_bbox = class_dct[Classes.CONTOUR]
	contour_mid_x, contour_mid_y = int((contour_bbox['x1'] + contour_bbox['x2']) // 2), int((contour_bbox['y1'] + contour_bbox['y2']) // 2)
	digits.sort(key=lambda x: x.value)
	twelve_ref = digits[-1]
	tweleve_ref_coord = digits_center_pos[twelve_ref]
	angles = []

	start_vec = [tweleve_ref_coord[0] - contour_mid_x, tweleve_ref_coord[1] - contour_mid_y]
	cum_sum_angle = 0
	for digit in digits:	# 1 - 11
		digit_coords = digits_center_pos[digit]
		digit_vec = [digit_coords[0] - contour_mid_x, digit_coords[1] - contour_mid_y]
		angle = compute_angle(start_vec, digit_vec)
		start_vec = digit_vec
		cum_sum_angle += angle
		angles.append(cum_sum_angle)
	#print(angles)
	if not all(angle < 361 for angle in angles):
		msg = "digits not in correct order"
		return 0, msg
	return 1, "passed check"

def add_contour(class_dct):
	# get all bboxes of digits
	digit_bboxes = [list(class_dct[cls].values()) for cls in class_dct if cls.value < 12]
	# find xmin,ymin,xmax,ymax
	xmin = min(list(map(lambda x: x[0], digit_bboxes)))
	xmax = max(list(map(lambda x: x[2], digit_bboxes)))
	ymin = min(list(map(lambda x: x[1], digit_bboxes)))
	ymax = max(list(map(lambda x: x[3], digit_bboxes)))

	# add contour
	class_dct[Classes.CONTOUR] = {'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2' : ymax}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--prediction_path", type=str, required=True, help="prediction file to score drawings")
	parser.add_argument("--test_images", type=str, help="dir to test images", default = "./data_v1.1/test/test_images")
	parser.add_argument("--hour", type=int, default=11)
	parser.add_argument("--im_draw", action='store_true', default=False)
	parser.add_argument("--minute", type=int, default =10)
	args = parser.parse_args()

	df = pd.read_csv(args.prediction_path)
	df = df[df['prob_score'] >= OBJ_THRESH]

	time = {}
	time['hour'] = args.hour
	time['minute'] = args.minute

	images_list = df.image.unique()
	for image_name in images_list:
		print(image_name)
		full_image_path = os.path.join(args.test_images, image_name)
		im_score = {"contour": 0, "hand": 0, "digit": 0}
		im_pred_df = df[df['image'] == image_name]
		image = Image.open(full_image_path)
		image = image.resize((800, 800))
		# convert csv into dictionary format
		class_dct = convert(im_pred_df)
		classes_pred = list(class_dct.keys())
		hasContour = Classes.CONTOUR in class_dct
		if hasContour:
			im_score["contour"] = 1

		if not hasContour:
			add_contour(class_dct)
		digit_score, err_digit_msg = score_digits(class_dct, image)
		# add in virtual contour, if digits are well positioned and scored.
		hand_score, err_hand_msg = score_hand(class_dct, image, time, im_draw=args.im_draw)
		im_score['hand'] = hand_score
		im_score['digit'] = digit_score
		print("contour : {}, digit: {}, hand: {}".format(im_score['contour'], digit_score, hand_score))
		print("error msg digit: {}, error msg hand: {}".format(err_digit_msg, err_hand_msg))
		if args.im_draw:
			plt.imshow(image)
			plt.show()
