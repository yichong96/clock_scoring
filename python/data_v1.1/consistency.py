import pandas as pd
import os
import glob
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_dir", type=str, required=True)
	parser.add_argument("--csv_path", type=str, required=True)
	args = parser.parse_args()

	image_names = glob.glob(args.image_dir + "/*.png")
	image_names = set(map(lambda x: x.split("/")[-1], image_names))
	csv_names = set(pd.read_csv(args.csv_path)['filename'].unique())

	for name in csv_names:
		if name not in image_names:
			print(name)




