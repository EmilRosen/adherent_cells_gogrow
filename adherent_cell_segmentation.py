import numpy as np;
import os;
import parse;
import datetime;

from functools import partial;

import json;
import csv;

import pyvips;

import split;

import tracking;
from tracking import utils;
from tracking import models;
from tracking import filter;
from tracking import segment;
from tracking import stitch;
from tracking import cell as cellModule;
from tracking import plot;
from tracking import weightMap as wm;

import incucyte_images;

from pprint import pprint;

from keras import callbacks;
from keras.preprocessing.image import ImageDataGenerator;

import scipy.misc;

from matplotlib import pyplot as plt;

from skimage import exposure;
from scipy.ndimage import generic_filter;

import cv2;

import pandas as pd;

from pathos.multiprocessing import ProcessingPool as Pool;

from skimage import measure;

from particle_math import pairCorrelation, particlesInRectangle;
from scipy.signal import savgol_filter;


###########
## Input ##
###########
dpi = 92;



basePath = "/media/emiro593/AdherentCells/";
rawPath = basePath + "Ludmila_2020_06_02/";

folderPattern  = "{plate:d}_{row}{col:02d}";
filePattern    = "{day:02d}d{hour:02d}h{minute:02d}m.{fileEnding}";

rawFolderPattern   = "{plate:d}";

rawPattern         = "VID{vid}_{row}{col:d}_1_{day:02d}d{hour:02d}h{minute:02d}m.{fileEnding}";
paintedPattern     = "{index:d}_{type}.{fileEnding}";

##
##
##
paintedPath    = basePath + "Painted/";
filterPath     = basePath + "Filtered/";
stabilizedPath = basePath + "Export_Stabilized/";
particlePath   = basePath + "Particle/"
summaryPath    = basePath + "Summary/"
baxterPath     = basePath + "Baxter/"
weightPath     = basePath + "WeightMap/";

track_path = basePath + "Track/";
track_debug_path = basePath + "DebugTrack/";

netPath = basePath + "model.h5";

numLayers = 4;
numFilters = 64;
epochs = 1000;

weights = [
	{"weight":  1  , "operation": "addition", "type": "weightMap_Balance", "optional": False, "padParams": {"mode": "edge"}},
	{"weight":  1.5, "operation": "addition", "type": "weightMap_Border" , "optional": False, "padParams": {"mode": "edge"}},
];

imageSize = 256;

# Incucyte sepcs for ludmila adherent experiment
# Image Resolution: 1.24 microns/pixel
# Image Size: 1408 x 1040 pixels
# Field of View: 1.75 x 1.29 mm

micronsPerPixel = 1.24;

# width X height
full_image_size = [1408, 1040];

# Standard diameter of 96 well plate is 6.94 mm --> 6940 um
# pi * (6940/2)^2 = 37827602.9826
# 1408 * 1040 * 1.24^2 = 2251538.432
# Ratio --> 2251538.432 / 37827602.9826 = 0.05952104427 ~ 6%
# 500 seeded --> 30 at start
# In rality we see between 20 - 50 cells at the start. Reasonable?

################
## Initialize ##
################
np.random.seed(5767);

utils.defineOutput(filterPath);
utils.defineOutput(particlePath);
utils.defineOutput(baxterPath);
utils.defineOutput(weightPath);
utils.defineOutput(stabilizedPath);
utils.defineOutput(summaryPath);
utils.defineOutput(track_path);
utils.defineOutput(track_debug_path);

classes = [
	{"name": "painted", "index": 0, "color": 0x000000},		# Background
	{"name": "painted", "index": 1, "color": 0xFF0000},		# Dead cells
	{"name": "painted", "index": 2, "color": 0xFFFF00},		# Censored
	{"name": "painted", "index": 3, "color": 0xFFFFFF},		# Cells
	{"name": "painted", "index": 4, "color": 0x00FF00},		# Protrusions
];

def equalize(I):
	return exposure.equalize_adapthist(I, clip_limit=0.02);

if __name__ == '__main__':
	
	###
	## Copy blah
	###
	'''
	from shutil import copyfile;

	inputs = incucyte_images.load(basePath + "RAW", rawPattern);
	inputs = sorted(inputs, key=lambda x: (x["plate"], x["col"], x["row"]));

	print("Applying to: " + str(len(inputs)) + " images");

	def copy_file_to_folder(input):
		#if input["row"] != "A":
		#	continue;

		savePath = basePath + "SortedRaw/" + folderPattern.format(**input) + "/";
		utils.defineOutput(savePath);

		saveFile = savePath + filePattern.format(**{**input});

		copyfile(input["path"] + "/" + input["file"], saveFile);

	with Pool(10) as pool:
		pool.map(copy_file_to_folder, inputs);

	a = b;
	'''
	#################################
	## Generate balanced weightmap ##
	#################################
	'''
	inputs  = utils.getInputFiles(paintedPattern, paintedPath, lambda input: input["type"] == "painted");
	
	numScores = {};
	total     = 0;
	for input in inputs:
		mask   = utils.loadImage(input["path"] + input["file"], "HEX");

		for row in classes:
			color = row["color"];
			index = row["index"];

			if index not in numScores.keys():
				numScores[index] = 0;
			
			numScores[index] += np.sum(mask == color); 
			total    	 	 += np.sum(mask == color);

	for input in inputs:
		mask      = utils.loadImage(input["path"] + input["file"], "HEX");
		weightMap = np.zeros(mask.shape);

		for row in classes:
			color = row["color"];
			index = row["index"];

			# TODO: Check for numScores == 0 / too small
			weightMap[mask == color] = np.min(list(numScores.values())) / numScores[index];

		meta = {**input, **{"type": "weightMap_Balance"}};
		scipy.misc.toimage(weightMap, cmin=0, cmax=1).save(weightPath + paintedPattern.format(**meta));
	
	###############################
	## Generate border weightmap ##
	###############################
	inputs  = utils.getInputFiles(paintedPattern, paintedPath, lambda input: input["type"] == "painted");
	inputs  = utils.filterDoneInput(inputs, weightPath, lambda input: paintedPattern.format(**{**input, "type": "weightMap_Border"}));

	for input in inputs:
		IO = utils.loadImage(input["path"] + input["file"], "HEX");
		IO = utils.hex2class(IO, classes);

		I = np.zeros(shape=(IO.shape[0], IO.shape[1], len(classes)));
		for n in range(len(classes)):
			I[:, :, n] = (IO == n); #mask = utils.loadBinaryImage(fileName).astype("uint8");

		I = I.astype("uint8");
		weightMap = wm.computeBorderWeightMap(I);

		#plt.imshow(weightMap);
		#plt.show();

		meta = {**input, **{"type": "weightMap_Border"}};
		scipy.misc.toimage(weightMap, cmin=0, cmax=1).save(weightPath + paintedPattern.format(**meta));
	'''
	###################
	## Train Network ##
	###################
	'''
	print("Generate / load neural network");

	tentacleNet = models.load(netPath);
	
	originalInputs = utils.getInputFiles(paintedPattern, paintedPath, lambda input: (input["type"] == "raw" and input["fileEnding"] != "pdn"));
	paintedInputs  = utils.getInputFiles(paintedPattern, paintedPath, lambda input: input["type"] == "painted");

	borderInputs   = utils.getInputFiles(paintedPattern, weightPath, lambda input: input["type"] == "weightMap_Border");
	balancedInputs = utils.getInputFiles(paintedPattern, weightPath, lambda input: input["type"] == "weightMap_Balance");

	inputs = utils.mergeInputs(["index"], {"original": originalInputs, "painted": paintedInputs, "weightMap_Balance": balancedInputs, "weightMap_Border": borderInputs});
	
	if not tentacleNet:
		print("Neural network not found, training new network...");

		trainingSet = utils.loadWeightedTrainingSamples(inputs, classes, weights);

		numSamples = trainingSet["inputs"].shape[0];
		print(numSamples);

		dataArgs = dict(
			rotation_range=180,
	        width_shift_range=0.2,
	        height_shift_range=0.2,
	        shear_range=0.2,
	        zoom_range=0.1,
	        horizontal_flip=True,
	        vertical_flip=True,
	        fill_mode='reflect');

		traingen  = ImageDataGenerator(**dataArgs);
		targetgen = ImageDataGenerator(**{**dataArgs, "fill_mode": "constant", "cval": 0});

		trainingSet["imageSize"] = (imageSize, imageSize, 1);

		# Data augmentation method
		def datagen(traingen, targetgen, batchSize, classes):
			seed = np.random.randint(0, 2**32);

			traingen  = traingen .flow(trainingSet["inputs"] , seed=seed, batch_size=batchSize);
			targetgen = targetgen.flow(trainingSet["targets"], seed=seed, batch_size=batchSize);

			for X, Y in zip(traingen, targetgen):				
				for i in range(X.shape[0]):
					X[i, :, :, 0] = equalize(X[i, :, :, 0]);

				yield X, Y.reshape(batchSize, -1, len(classes) + 1);

		tentacleNet = models.createUNET(trainingSet['imageSize'], numLayers, numFilters, classes=len(classes), kernel_size=(3, 3));

		batchSize  = 5;
		savePeriod = 50;

		saveCheckpoint    = callbacks.ModelCheckpoint(netPath, verbose=0, period=savePeriod, save_best_only=True, monitor="loss", mode="min");

		tentacleNet.fit_generator(datagen(traingen, targetgen, batchSize, classes), steps_per_epoch=(5 * numSamples / batchSize), epochs=epochs, verbose=1, callbacks=[saveCheckpoint]);

		tentacleNet = models.load(netPath);
	'''
	
	###################
	## Filter Images ##
	###################
	'''
	def parse_platemap(xml, input):
		row = ord(input["row"]) - 65;
		col = input["col"] - 1;

		for well in xml.iter("well"):
			if int(well.attrib["row"]) == row and int(well.attrib["col"]) == col:
				if len(well) > 0:
					celline        = well.find("items/wellItem/referenceItem").attrib["displayName"];
					seedingDensity = well.find("items/wellItem").attrib["seedingDensity"];
					
					return {"row": input["row"], "col": input["col"], "celline": int(celline[:4]), "density": int(seedingDensity)};

		print("Error: No data for {}{}".format(input["row"], input["col"]));

		return {};

	import xml.etree.ElementTree as ET;
	'''
	'''
	print("Apply neural network filter");

	#parentInputs = utils.getInputFiles(rawFolderPattern, rawPath, filter = lambda row: row["plate"] in [106, 108]);
	parentInputs = utils.getInputFiles(rawFolderPattern, rawPath);

	for parentInput in parentInputs:
		#xml_root = ET.parse(basePath + "PlateMap/{plate} {name}.PlateMap".format(**parentInput)).getroot();

		inputs = utils.getInputFiles(rawPattern, parentInput["path"] + "/" + parentInput["file"] + "/");

		for input in inputs:
			#xml_dict = parse_platemap(xml_root, input);

			#if xml_dict["density"] in [10000, 5000]:
			#	continue;

			#savePath = filterPath + folderPattern.format(**{**parentInput, **xml_dict}) + "/";
			savePath = filterPath + folderPattern.format(**{**parentInput, **input}) + "/";
			utils.defineOutput(savePath);
			
			saveFile = savePath + filePattern.format(**{**input, "fileEnding": "png"});

			if not os.path.isfile(saveFile):
				R = pyvips.Image.new_from_file(input["path"] + "/" + input["file"]);
				R = np.ndarray(buffer=R.write_to_memory(), dtype=np.uint8, shape=[R.height, R.width, R.bands]);
				R = R.astype("float32") / 255;

				R = equalize(R[:, :, 0]);

				FS = [];
				for I, p in split.splitImage(R, imageSize, 48):					
					I = {"image": I, "paddings": [[0, 0], [0, 0]]};

					F = filter.filterImage(I, tentacleNet, len(classes));

					FS.append({"offset": p, "image": F});

				F = split.fusePatches(FS, classes);
				L = utils.layered2rgb(F, classes);
							
				scipy.misc.toimage(L, cmin=0, cmax=1).save(saveFile);
	'''
	#######################
	## Images for baxter ##
	#######################
	'''
	parentInputs = utils.getInputFiles(folderPattern, filterPath);
	
	for parentInput in parentInputs:
		savePath = baxterPath + folderPattern.format(**parentInput) + "/";
		utils.defineOutput(savePath);

		inputs = utils.getInputFiles(filePattern, parentInput["path"] + parentInput["file"]);

		for input in inputs:
			saveFile = savePath + filePattern.format(**{**input, "fileEnding": "png"});
			
			if not os.path.isfile(saveFile):
				I = utils.loadImage(input["path"] + "/" + input["file"], type="HEX");

				I2 = np.zeros(shape=(I.shape[0], I.shape[1]));
				I2[I == classes[3]["color"]] = 1;

				scipy.misc.toimage(I2, cmin=0, cmax=1).save(saveFile);
	
	'''
	#######################
	## Analyze Particles ##
	#######################
	'''
	print("Analyzing particles");

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));

	#parentInputs = utils.getInputFiles(folderPattern, filterPath, filter=lambda row: row["col"] not in [2, 3, 5, 6, 7, 8, 10, 11]);
	parentInputs = utils.getInputFiles(folderPattern, filterPath);

	for parentInput in parentInputs:
		inputs = utils.getInputFiles(filePattern, parentInput["path"] + parentInput["file"]);
		#inputs = utils.getInputFiles(filePattern, parentInput["path"] + parentInput["file"], filter=lambda row: row["day"] == 0 and row["hour"] in [0] and row["minute"] in [0]);

		savePath = particlePath + folderPattern.format(**parentInput) + "/";
		utils.defineOutput(savePath);

		inputs = utils.filterDoneInput(inputs, savePath, lambda input: filePattern.format(**{**input, "fileEnding": "json"}));

		for input in inputs:
			print(parentInput["file"], input["file"]);

			I = utils.loadImage(input["path"] + "/" + input["file"], "HEX");
			I = utils.hex2layered(I, classes);
			
			I = I[3];

			I = (I * 255).astype("uint8");			
			
			I = cv2.morphologyEx(I, cv2.MORPH_OPEN, kernel, iterations=2);
			B = cv2.dilate(I, kernel, iterations=4);

			D = cv2.distanceTransform(I, cv2.DIST_L2, 5);

			ret, F = cv2.threshold(D, 2, 255, 0);

			U = B - F;

			ret, M = cv2.connectedComponents(F.astype("uint8"), 4);
			M += 1;

			M[U == 255] = 0;

			M = cv2.watershed(np.dstack((I, I, I)), M);

			#plt.imshow(M);
			#plt.show();

			props     = measure.regionprops(M);
			
			particles = [];
			for prop in props:
				r = np.sqrt(prop.area / np.pi) * micronsPerPixel;

				if 2 * r > 15 and 2 * r < 100:
					particles.append({"X": float(prop.centroid[1] * micronsPerPixel), "Y": float(prop.centroid[0] * micronsPerPixel), "radius": r});

			col2seed = {2: 2000, 3: 1000, 4: 500, 5: 250, 6: 125, 7: 2000, 8: 1000, 9: 500, 10: 250, 11: 125};
			#print(parentInput["file"], input["day"], len(particles), col2seed[parentInput["col"]]);

			with open(savePath + filePattern.format(**{**input, "fileEnding": "json"}), 'w') as fp:
				json.dump(particles, fp);
	'''
	################
	## Confluence ##
	################
	'''
	print("Measuring confluence");

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));

	parentInputs = utils.getInputFiles(folderPattern, filterPath, filter=lambda row: row["plate"] in [106, 108] and row["density"] not in [10000, 5000]);

	for parentInput in parentInputs:
		inputs = utils.getInputFiles(filePattern, parentInput["path"] + parentInput["file"], filter=lambda row: row["day"] == 4 and row["hour"] in [0, 1] and row["minute"] in [0, 20]);

		savePath = particlePath + folderPattern.format(**parentInput) + "/";
		utils.defineOutput(savePath);

		#inputs = utils.filterDoneInput(inputs, savePath, lambda input: filePattern.format(**{**input, "fileEnding": "json"}));

		for input in inputs:
			print(parentInput["file"], input["file"]);

			I = utils.loadImage(input["path"] + "/" + input["file"], "HEX");
			I = utils.hex2layered(I, classes);
			
			I = I[3];

			I = (I * 255).astype("uint8");			
			
			I = cv2.morphologyEx(I, cv2.MORPH_OPEN, kernel, iterations=2);
			B = cv2.dilate(I, kernel, iterations=4);

			D = cv2.distanceTransform(I, cv2.DIST_L2, 5);

			ret, F = cv2.threshold(D, 2, 255, 0);

			U = B - F;

			ret, M = cv2.connectedComponents(F.astype("uint8"), 4);
			M += 1;

			M[U == 255] = 0;

			M = cv2.watershed(np.dstack((I, I, I)), M);

			props     = measure.regionprops(M);
			
			particles = [];
			for prop in props:
				r = np.sqrt(prop.area / np.pi) * micronsPerPixel;

				if 2 * r > 15 and 2 * r < 100:
					particles.append({"X": float(prop.centroid[1] * micronsPerPixel), "Y": float(prop.centroid[0] * micronsPerPixel), "radius": r});

			with open(savePath + filePattern.format(**{**input, "fileEnding": "json"}), 'w') as fp:
				json.dump(particles, fp);
	'''
	'''
	##############################
	## Confluence: No particles ##
	##############################
	print("Measuring confluence");

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));

	parentInputs = utils.getInputFiles(folderPattern, filterPath);

	#parentInputs = np.random.choice(parentInputs, len(parentInputs), replace=False);

	for parentInput in parentInputs:
		inputs = utils.getInputFiles(filePattern, parentInput["path"] + parentInput["file"]);

		savePath = particlePath + folderPattern.format(**parentInput) + "/";
		utils.defineOutput(savePath);

		if parentInput["row"] != "B":
			continue;

		well = [];

		for input in inputs:
			I = utils.loadImage(input["path"] + "/" + input["file"], "HEX");
			I = utils.hex2layered(I, classes);
			
			time = input["day"] * 24 + input["hour"];
			S = np.sum(I[3]);

			well.append({"Time": time, "Confluence": S});

		well = pd.DataFrame(well);
		well = well.sort_values("Time");

		plt.plot(well["Time"], well["Confluence"]);
		plt.title(parentInput["file"]);
		plt.show();

	a = b;
	'''
	########################
	## Summary statistics ##
	########################
	'''
	print("Extract summary statistics");

	parentInputs = utils.getInputFiles(folderPattern, particlePath);

	baseRadius = 20;
	maxRadius  = 260;

	required_cells = 50;

	radii = None;

	for parentInput in parentInputs:
		inputs = utils.getInputFiles(filePattern, parentInput["path"] + parentInput["file"], filter=lambda row: row["day"] == 4 and row["hour"] in [0]);

		imsize = [full_image_size[0] * micronsPerPixel, full_image_size[1] * micronsPerPixel];
		#side   = full_image_size[0] * micronsPerPixel / 4;

		# x, y, w, h
		#rect = [imsize[0] / 2 - side, imsize[1] / 2 - side, imsize[0] / 2 + side, imsize[1] / 2 + side];
		rect = [0, 0, imsize[0], imsize[1]];

		sizes        = [];
		numParticles = [];
		radial       = [];

		data = [];

		for input in inputs:
			print(parentInput["file"], input["file"]);

			with open(input["path"] + "/" + input["file"]) as fp:
				particles = json.load(fp);

				P = [[p["X"], p["Y"]] for p in particles];
				S = [p["radius"] for p in particles];

				if len(P) <= 0:
					continue;

				N = int(particlesInRectangle(np.array(P), rect));

				#if N < required_cells:
				#	continue;

				numParticles.append(N);
				sizes = sizes + S;
				
				pc, radii = pairCorrelation(np.array(P), rect, maxRadius, baseRadius);

				if False:
					print("Num cells:", "{plate} {row}{col}".format(**{**input, **parentInput}), " : ", N, np.mean(sizes));
					plt.figure(figsize=(10, 10));
					plt.subplot(2, 1, 1);
					plt.suptitle("{plate} {row}{col}".format(**{**input, **parentInput}));
					plt.scatter(np.array(P)[:, 0], np.array(P)[:, 1]);
					plt.plot([rect[0], rect[0]], [rect[1], rect[3]], "k--");
					plt.plot([rect[2], rect[2]], [rect[1], rect[3]], "k--");
					plt.plot([rect[0], rect[2]], [rect[1], rect[1]], "k--");
					plt.plot([rect[0], rect[2]], [rect[3], rect[3]], "k--");
					plt.subplot(2, 1, 2);
					plt.plot(radii, pc);
					plt.show();
				
				data.append({"plate": parentInput["plate"], "col": parentInput["col"], "row": parentInput["row"], "num": N, "pairCorrelation": list(pc), "radii": list(radii), "dr": baseRadius, "rMax": maxRadius, "fov": [imsize[0], imsize[1]]});

				if not np.any(np.isnan(pc)):
					radial.append(pc);

		counts, bins, whatevs = plt.hist(sizes, bins=100); plt.close("all");
		smoothedCounts = savgol_filter(counts, 41, 3);

		particleSize = (bins[np.argmax(smoothedCounts)] + bins[np.argmax(smoothedCounts) + 1]) / 2;
		#numParticles = np.median(numParticles);

		#radial = np.mean(np.array(radial), axis=0);

		#plt.plot(radii, radial);
		#plt.plot([particleSize * 2, particleSize * 2], [0, np.max(radial)]);
		#plt.show();

		for i, row in enumerate(data):
			data[i]["radius"] = particleSize;

		if len(data) > 0:
			with open(summaryPath + folderPattern.format(**parentInput) + ".json", 'w') as fp:
				json.dump(data, fp);

	
	a = b;
	'''

	#############
	## Trackpy ##
	#############
	import trackpy;

	print("Trackpy tracking");

	parentInputs = utils.getInputFiles(folderPattern, particlePath);

	def track(parentInput):
		inputs = utils.getInputFiles(filePattern, parentInput["path"] + parentInput["file"]);

		save_file = track_path + folderPattern.format(**parentInput) + ".csv";

		df = [];

		time2date = {};

		parentInput["vid"]        = parentInput["plate"];
		parentInput["fileEnding"] = "tif";

		for input in inputs:
			with open(input["path"] + "/" + input["file"]) as fp:
				particles = json.load(fp);

				time = input["day"] * 24 + input["hour"];

				df_ = pd.DataFrame(particles);
				df_["Time"] = time;

				time2date[time] = {"day": input["day"], "hour": input["hour"], "minute": input["minute"]};

				df.append(df_);

		df = pd.concat(df);
		df = df.rename(columns={"X": "x", "Y": "y", "Time": "frame"});

		print(parentInput["file"]);
		
		tracks = trackpy.link_df(df, 300, memory=1, adaptive_step=0.95, adaptive_stop=1);

		tracks.to_csv(save_file, index=False, sep="\t");

		##
		## Plot debug tracks
		if False:
			savePath = track_debug_path + folderPattern.format(**parentInput) + "/";
			utils.defineOutput(savePath);

			tt = sorted(list(tracks["frame"].unique()));

			for t in tt:
				frame = tracks[tracks["frame"] == t];
				ids = frame["particle"];

				full = tracks[tracks["particle"].isin(ids)];

				plt.figure(figsize=(8, 8));

				import imageio;

				raw_impath = rawPath    + rawFolderPattern.format(**parentInput) + "/" + rawPattern.format(**{**parentInput, **time2date[t]});
				fil_impath = filterPath + folderPattern.format(**parentInput) + "/" + filePattern.format(**{**parentInput, **time2date[t], "fileEnding": "png"});

				I = imageio.imread(fil_impath);

				plt.imshow(I, cmap='Greys');

				for id in ids:
					track = full[(full["particle"] == id) & (full["frame"] <= t) & (full["frame"] >= t - 10)];

					track["x"] = track["x"] / micronsPerPixel;
					track["y"] = track["y"] / micronsPerPixel;

					if len(track) == 1:
						plt.scatter(track["x"], track["y"], s=4);
					else:
						plt.plot(track["x"], track["y"]);

				plt.xlim([0, full_image_size[0]]);
				plt.ylim([0, full_image_size[1]]);

				plt.savefig(savePath + filePattern.format(**{**parentInput, **time2date[t], "fileEnding": "png"}) + "");

				plt.close("all");

				if t > 40:
					break;

	pool = Pool(nodes=10);
	pool.map(track, parentInputs);

	
