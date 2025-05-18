#!/usr/bin/env python3

import os
import torch
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse
import logging

from models.config import _C as cfg
from models.networks import MiDas, OursSSI, OursSI, BMDSSI
from models.bmd.midas.utils import write_depth

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

# Load the image and convert it to RGB
def load_image(img_filename):
	img = np.asarray(Image.open(img_filename).convert('RGB'))
	return img

# Save the depth image
def save_normalized_depth_image(model_name, output_dir_tif, path_name, image):
		rgb_name_base = os.path.splitext(os.path.basename(path_name))[0]
		pred_name_base = rgb_name_base + "_pred"
		png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
		if os.path.exists(png_save_path):
			logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
		write_depth(os.path.join(png_save_path), image, colored=False)

if "__main__" == __name__:
	logging.basicConfig(level=logging.INFO)

	parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using DPT_Large or MiDaS_small."
    	)

	parser.add_argument(
	"-i", "--input_rgb_dir",
	type=str,
	required=True,
	help="Path to the input image folder.",
	)

	parser.add_argument(
	"-o", "--output_dir",
	type=str,
	required=True,
	help="Output directory."
	)

	parser.add_argument(
	"-m", "--model_name", choices = ['MiDaS', 'DPT', 'SSI', 'SI' , 'SSIBase'],
	help="Prediction model (MiDaS_small, DPT_Large, SSI, SI, SSIBase",
	)

	args = parser.parse_args()

	if args.model_name == "MiDaS" or args.model_name == "DPT":
		model = MiDas(args.model_name)	
	elif args.model_name == "SSI":
		model = BMDSSI(cfg.model_config)
	elif args.model_name == "SSIBase":
		model = OursSSI(cfg.model_config)
	elif args.model_name == "SI":
		model = OursSI(cfg.model_config)
	
	input_rgb_dir = args.input_rgb_dir
	output_dir = args.output_dir
	cfg.model_config = args.model_name
	

	output_dir_tif = os.path.join(output_dir, "depth_bw")
	os.makedirs(output_dir_tif, exist_ok=True)
	logging.info(f"output dir = {output_dir}")

	rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
	rgb_filename_list = [
		f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
	]
	rgb_filename_list = sorted(rgb_filename_list)
	n_images = len(rgb_filename_list)
	if n_images > 0:
		logging.info(f"Found {n_images} images")
	else:
		logging.error(f"No image found in '{input_rgb_dir}'")
		exit(1)

	with torch.no_grad():
		os.makedirs(output_dir, exist_ok=True)

		items = n_images - 2
		first = rgb_filename_list[0]
		last = rgb_filename_list[n_images - 1]

		# Process the first image
		# and save it to the output directory
		# with the same name as the input image
		# but with the "_pred" suffix
		
		img = load_image(first)
		prediction = model.evaluate(img)
		save_normalized_depth_image(args.model_name, output_dir_tif, first, prediction)

		# Process the second image with averaging
		# and save it to the output directory
		arr = np.zeros((prediction.shape[0], prediction.shape[1]), np.float64)
		prev = prediction
		
		curr = model.evaluate(load_image(rgb_filename_list[1]))
		next = model.evaluate(load_image(rgb_filename_list[2]))

		arr = arr + prev / 3
		arr = arr + curr / 3
		arr = arr + next / 3
		
		save_normalized_depth_image(args.model_name, output_dir_tif, rgb_filename_list[1], arr)
		
		# Process all remaining but the last images with averaging
		# and save them to the output directory
		 
		current	= 2

		for idx in tqdm(range(items - 1)):
			current = idx + 2
			arr = np.zeros((prediction.shape[0], prediction.shape[1]), np.float64)
			prev = curr
			curr = next
			next = model.evaluate(load_image(rgb_filename_list[current + 1]))

			arr = arr + prev / 3
			arr = arr + curr / 3
			arr = arr + next / 3
						
			save_normalized_depth_image(args.model_name, output_dir_tif, rgb_filename_list[current], arr)
	
		# Process the last image
		# and save it to the output directory
			
		prediction = model.evaluate(load_image(last))
		save_normalized_depth_image(args.model_name, output_dir_tif, last, prediction)
			
	print('Done.')
		
