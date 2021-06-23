import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--main_xml", type=str, required=True)
	parser.add_argument("--xml_to_add", type=str, required=True)
	parser.add_argument("--save_dir", type=str, required=True)

	args = parser.parse_args()
	main_dir = args.main_xml
	xml_to_add = args.xml_to_add
	save_dir = args.save_dir

	main_files = glob.glob(main_dir + "/*.xml")
	sub_files = glob.glob(xml_to_add + "/*.xml")

	#main_files = list(map(lambda x: x.split("/")[-1], main_files))
	#sub_files = list(map(lambda x: x.split("/")[-1], sub_files))
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	for file in main_files:
		xml_file = ET.parse(file)
		for sub_file in sub_files:
			main_name = file.split("/")[-1]
			sub_name = sub_file.split("/")[-1]
			if main_name == sub_name:
				xml_sub_file = ET.parse(sub_file)
				main_root = xml_file.getroot()
				sub_root = xml_sub_file.getroot()
				sub_objs = sub_root.findall("object")
				for obj in sub_objs:
					main_root.append(obj)
				break
		xml_file.write(os.path.join(save_dir, main_name))