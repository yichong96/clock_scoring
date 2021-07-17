import pandas as pd
import os
import argparse
import math
import numpy as np
import shutil
import glob
"""
	80-20 split between training and testing
"""

"""
	appends images and xml file into respective train and test lists for each diagnosis.
"""
def gather(images_lst, xml_lst, indices, train_images, test_images, train_xml, test_xml):
	for i in range(len(images_lst)):
		if i in indices:
			train_images.append(images_lst[i])
			train_xml.append(xml_lst[i])
		else:
			test_images.append(images_lst[i])
			test_xml.append(xml_lst[i])


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_dir", type=str, default = "./all_data")
	parser.add_argument("--xml_dir", type=str, default="./all_data_xml")
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	data_dir = args.image_dir
	xml_dir = args.xml_dir
	seed = args.seed

	# reproducible test and train set
	np.random.seed(0)

	image_lst = glob.glob(data_dir + "/*.png")
	xml_lst = glob.glob(xml_dir + "/*.xml")


	xml_set = set(list(map(lambda x: x.split("/")[-1].split(".")[0], xml_lst)))
	image_set = set(list(map(lambda x: x.split("/")[-1].split(".")[0], image_lst)))
	# some images not used in training
	set_intersection = xml_set.intersection(image_set)


	image_lst = list(filter(lambda x: x.split("/")[-1].split(".")[0] in set_intersection, image_lst))

	NCI_image_lst = sorted(list(filter(lambda x: "NCI" in x, image_lst)))
	NCI_xml_lst = sorted(list(filter(lambda x: "NCI" in x, xml_lst)))

	MCI_image_lst = sorted(list(filter(lambda x: "MCI" in x, image_lst)))
	MCI_xml_lst = sorted(list(filter(lambda x: "MCI" in x, xml_lst)))

	MiD_image_lst = sorted(list(filter(lambda x: "MiD" in x, image_lst)))
	MiD_xml_lst = sorted(list(filter(lambda x: "MiD" in x, xml_lst)))

	MD_image_lst = sorted(list(filter(lambda x: "MD" in x, image_lst)))
	MD_xml_lst = sorted(list(filter(lambda x: "MD" in x, xml_lst)))

	NCI_indices = np.random.choice(len(NCI_image_lst), math.floor(0.8 * len(NCI_image_lst)), replace=False)
	MCI_indices = np.random.choice(len(MCI_image_lst), math.floor(0.8 * len(MCI_image_lst)), replace=False)
	MiD_indices = np.random.choice(len(MiD_image_lst), math.floor(0.8 * len(MiD_image_lst)), replace=False)
	MD_indices = np.random.choice(len(MD_image_lst), math.floor(0.8 * len(MD_image_lst)), replace=False)

	train_images = []
	train_xml = []

	test_images = []
	test_xml = []

	gather(NCI_image_lst, NCI_xml_lst, NCI_indices, train_images, test_images, train_xml, test_xml)
	gather(MCI_image_lst, MCI_xml_lst, MCI_indices, train_images, test_images, train_xml, test_xml)
	gather(MiD_image_lst, MiD_xml_lst, MiD_indices, train_images, test_images, train_xml, test_xml)
	gather(MD_image_lst, MD_xml_lst, MD_indices, train_images, test_images, train_xml, test_xml)

	train_folder = os.path.join(os.getcwd(), "train")
	test_folder = os.path.join(os.getcwd(), "test")

	try:
		os.mkdir(train_folder)
		os.mkdir(test_folder)
	except FileExistsError as e:
		pass

	train_image_folder = os.path.join(train_folder, "train_images")
	train_xml_folder = os.path.join(train_folder, "train_xml")
	test_image_folder = os.path.join(test_folder, "test_images")
	test_xml_folder = os.path.join(test_folder, "test_xml")

	try:
		os.mkdir(train_image_folder)
		os.mkdir(train_xml_folder)
		os.mkdir(test_image_folder)
		os.mkdir(test_xml_folder)
	except FileExistsError as e:
		pass

	# copy files from one dir to another
	for filename in train_images:
		shutil.copy(filename, train_image_folder)
	for filename in train_xml:
		shutil.copy(filename, train_xml_folder)
	for filename in test_images:
		shutil.copy(filename, test_image_folder)
	for filename in test_xml:
		shutil.copy(filename, test_xml_folder)




