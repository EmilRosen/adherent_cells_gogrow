import datetime;

import numpy as np;
import pandas as pd;


from matplotlib import pyplot as plt;
import seaborn;

from tracking import utils;
import incucyte_images;

import pyvips;

import scipy.misc;
from scipy.spatial.distance import cdist;
from scipy.io import loadmat;

from scipy.optimize import curve_fit;

##
##
##
basePath = "/media/emiro593/AdherentCells/";

trackPath = basePath + "Tracks/";
debugPath = basePath + "Debug/";


trackPattern  = "{plate:d}_{row}{col:02d}_{celline}_{density}.{fileEnding}";
folderPattern  = "{plate:d}_{row}{col:02d}_{celline:d}_{density:d}";
filePattern    = "{day:02d}d{hour:02d}h{minute:02d}m.{fileEnding}";

##
##
##
celline2motile = {
	3051: 0,
	3230: 1,
	3035: 1,
	3017: np.nan,
	3053: 0,
	3016: 0,
	3013: 0,
	3054: 0,
	3118: 1,
	3180: 1
};

def well2celline(input):
	if input["plate"] == 285:
		low  = 3051;
		high = 3230;
	if input["plate"] == 286:
		low  = 3035;
		high = 3017;
	if input["plate"] == 287:
		low  = 3053;
		high = 3016;
	if input["plate"] == 288:
		low  = 3013;
		high = 3054;
	if input["plate"] == 289:
		low  = 3118;
		high = 3180;

	return high if input["col"] >= 7 else low;

def well2density(input):
	densityIndex = input["col"] if input["col"] < 7 else input["col"] - 6;

	return 2000 / 2**(densityIndex - 1);

##
##
##
def findAttraction(X, Y):
	P1 = None;
	D1 = None;

	ndistances = [];
	ddistances = [];

	for t in range(X.shape[1]):
		P0 = P1;
		P1 = np.stack((X[:, t], Y[:, t])).T;

		D0 = D1;
		D1 = cdist(P1, P1);
		
		np.fill_diagonal(D1, np.nan);
		D1[np.isnan(D1)] = np.inf;

		if t > 0:
			n1 = np.arange(X.shape[0]);
			n2 = np.argmin(D0, axis=1);

			DD = D1 - D0;

			ND = D0[n1, n2];
			DD = DD[n1, n2];

			K = np.logical_and(np.abs(ND) < 200, np.abs(DD) < 50);

			ND = ND[K];
			DD = DD[K];

			ndistances = ndistances + list(ND);
			ddistances = ddistances + list(DD);

	return ndistances, ddistances;

def distanceJump(X, Y):
	dX = np.diff(X, axis=1);
	dY = np.diff(Y, axis=1);
	
	dP = np.sqrt(dX**2 + dY**2);

	ndistances = [];
	jumps      = [];

	for t in range(0, X.shape[1] - 1):
		P = np.stack((X[:, t], Y[:, t])).T;
		D = cdist(P, P);
		
		np.fill_diagonal(D, np.nan);
		D[np.isnan(D)] = np.inf;

		n1 = np.arange(X.shape[0]);
		n2 = np.argmin(D, axis=1);

		ND = D[n1, n2];
		PS = dP[:, t];

		K = np.logical_and(np.abs(ND) < 200, np.abs(PS) < 50);

		ND = ND[K];
		PS = PS[K];

		ndistances = ndistances + list(ND);
		jumps      = jumps      + list(PS);

	return ndistances, jumps;

def findIndividualDiffusion(X, Y, trackLengthThreshold=15):
	OK = np.sum(1 - np.isnan(X), axis=1) > trackLengthThreshold;

	X = X[OK, :];
	Y = Y[OK, :];

	p = np.zeros(shape=X.shape);

	for tau in range(X.shape[1]):
		X0 = X if tau == 0 else X[:, :-tau];
		Y0 = Y if tau == 0 else Y[:, :-tau];

		px = np.nanmean((X[:, tau:] - X0)**2, axis=1);
		py = np.nanmean((Y[:, tau:] - Y0)**2, axis=1);

		tp = px + py;

		p[:, tau] = tp;

	T = np.matlib.repmat(np.arange(0, X.shape[1]), X.shape[0], 1) * 45 * 60;

	#D = p[:, 1:] / (4 * T[:, 1:]);
	#return D;

	return p;


###########
## Debug ##
###########
'''
def findRawImage(input, t, raws):
	tests = ["row", "plate", "col"]; #"year", "month", "day", "hour", "minute"];

	targetTimestamp = datetime.datetime(2018, 6, 18, 16, 38).timestamp() + 45 * 60 * t;

	for raw in raws:
		timestamp = datetime.datetime(raw["year"], raw["month"], raw["day"], raw["hour"], raw["minute"]).timestamp();

		allOk = targetTimestamp == timestamp;

		for test in tests:
			if raw[test] != input[test]:
				allOk = False;
				break;

		if allOk:
			return raw;

	return None;

raws = incucyte_images.load(basePath + "RAW", rawPattern);

inputs  = utils.getInputFiles(trackPattern, trackPath, lambda input: input["dimension"] == "X");

for input in inputs:
	X = np.loadtxt(input["path"] + trackPattern.format(**{**input, "dimension": "X"}));
	Y = np.loadtxt(input["path"] + trackPattern.format(**{**input, "dimension": "Y"}));

	savePath = debugPath + folderPattern.format(**input);
	utils.defineOutput(savePath);

	if X.ndim > 1:
		for t in range(X.shape[1]):
			raw = findRawImage(input, t, raws);

			if raw is not None:
				R = pyvips.Image.new_from_file(raw["path"] + "/" + raw["file"]);
				R = np.ndarray(buffer=R.write_to_memory(), dtype=np.uint8, shape=[R.height, R.width, R.bands]);
				
				I = R.astype("float32") / 255;
				I = np.squeeze(I);

				my_dpi = 92;
				plt.figure(figsize=(I.shape[1] / my_dpi, I.shape[0] / my_dpi), dpi=my_dpi);
				plt.imshow(I, vmin=0, vmax=1);
				plt.subplots_adjust(0, 0, 1, 1);

				plt.scatter(X[:, t], Y[:, t], color="k");
				for n in range(X.shape[0]):
					if ~np.isnan(X[n, t]):
						start = np.maximum(0, t - 10);
				
						plt.plot(X[n, start:(t + 1)], Y[n, start:(t + 1)], color="k");

				plt.xlim([0, I.shape[1]]);
				plt.ylim([0, I.shape[0]]);

				plt.axis("off");
				plt.savefig(savePath + "/" + str(t) + "_" + filePattern.format(**{**raw, "fileEnding": "png"}), bbox_inches='tight');

				plt.close();

a = b;
'''
##
##
##
inputs  = utils.getInputFiles(trackPattern, trackPath, lambda input: input["dimension"] == "X");

groups = {};
for input in inputs:
	celline = well2celline(input);
	density = well2density(input);

	if density != 500:
		continue;

	if celline not in groups:
		groups[celline] = [];

	groups[celline].append(input);

rows = [];
for inputs in groups.values():
	DS = [];

	cellnums = [];

	for input in inputs:
		celline = well2celline(input);
		density = well2density(input);

		X = np.loadtxt(input["path"] + trackPattern.format(**{**input, "dimension": "X"}));
		Y = np.loadtxt(input["path"] + trackPattern.format(**{**input, "dimension": "Y"}));

		X = X*1.24;
		Y = Y*1.24;

		if X.ndim > 1:
			initialCells = np.sum(1 - np.isnan(X[:, 0:3])) / 3;
			finalCells   = np.sum(1 - np.isnan(X[:, -3:])) / 3;

			cellnums.append([initialCells, finalCells]);

			D = findIndividualDiffusion(X, Y, 15);

			DS.append(D);

	cellnums = np.array(cellnums);

	T = np.arange(0, D.shape[1]) * 45 * 60;
	D = np.concatenate(DS, axis=0);

	plt.figure(figsize=(8, 8));
	plt.title("{} {}".format(celline, celline2motile[celline]));
	for n in range(D.shape[0]):
		plt.plot(T / 3600, D[n, :]);
	plt.xlabel("Lag time (h)");
	plt.ylabel("MSD");
	plt.savefig(basePath + "Plots/" + str(celline));
	#plt.show();

	#DM = np.nanmean(D, axis=1);
	#DMEAN = np.nanmedian(D[7:]);

	diffs = [];
	for i in range(D.shape[0]):
		OK = ~np.isnan(D[i, :]);
		TC = T[OK];
		DC = D[i, OK];

		popt, pcov = curve_fit(lambda x, D: D * x, TC, DC);

		diffs.append(popt[0] / 2); 

	#plt.hist(diffs, bins=20);
	#plt.show();


	#H, E = np.histogram(DM, bins=20, density=True);
	#E = (E[1:] + E[:-1]) / 2;

	#def exp(x, k):
	#	return k * np.exp(-x * k);
	
	#popt, pcov = curve_fit(exp, E, H);

	#plt.figure(figsize=(8, 8));
	#plt.title("{} {}".format(celline, celline2motile[celline]));
	#plt.hist(DM, bins=30, density=True);
	#plt.plot(E, exp(E, *popt));
	#plt.show();

	def compDT(cellnums):
		R = np.mean(cellnums[:, 1]) / np.mean(cellnums[:, 0]);
		T = (45 * 52 / 60);
		L = R / T;

		return np.log(2) / L;



	rows.append({
		"Celline"        : celline,
		#"Density"        : density,
		"IsMotile"		 : celline2motile[celline],
		#"S"				 : popt[0],
		#"D"				 : DMEAN,
		"D"				 : np.median(diffs),
		#"IC"             : np.mean(cellnums[:, 0]),
		#"FC"             : np.mean(cellnums[:, 1]),
		#"RC"             : np.mean(cellnums[:, 1]) / np.mean(cellnums[:, 0]),
		"DT"			 : compDT(cellnums)
	});

df = pd.DataFrame(rows);

from scipy.stats import pearsonr;
from scipy.stats import spearmanr;

plt.close("all");

'''
print(pearsonr(df["DT"], df["D"]));
print(spearmanr(df["DT"], df["D"]));

df.plot.scatter(x="DT", y="D");
for i, row in df.iterrows():
    plt.text(row["DT"], row["D"], row["Celline"]);

plt.xlabel("Doubling time (h)");
plt.ylabel("Diffusion coefficient (microns^2/s)");

plt.show();

df.plot.bar(x="Celline", y="D");
plt.ylabel("Diffusion coefficient (microns^2 / s)");
plt.show();
'''

dsdf = pd.read_csv(basePath + "DrugScreenMigrationParameters.csv", sep="\t");
dsdf["Celline"] = dsdf["Celline"].apply(lambda row: int(row[1:]));


dsdf = dsdf.merge(df, how="inner", on="Celline");
dsdf["DT_DS"] = np.log(2) / 10**dsdf["Proliferation"] / 3600;

plt.scatter(dsdf["DT_DS"], dsdf["DT"]);
for i, row in dsdf.iterrows():
    plt.text(row["DT_DS"], row["DT"], row["Celline"]);

plt.xlabel("Drug screen doubling time");
plt.ylabel("Adherent doubling time");
plt.show();


#g = seaborn.PairGrid(df, diag_sharey=False);
#g.map_lower(seaborn.kdeplot);
#g.map_upper(seaborn.scatterplot);
#g.map_diag(seaborn.kdeplot, lw=3);
#plt.show();

a = b;

##
##
##
inputs  = utils.getInputFiles(trackPattern, trackPath, lambda input: input["dimension"] == "X");

rows        = [];
jumpRows    = [];
attractRows = [];
disJumpRows = [];

msdRows     = [];

for input in inputs:
	#print(input);

	celline = well2celline(input);
	density = well2density(input);

	if density != 1000:
		continue;

	X = np.loadtxt(input["path"] + trackPattern.format(**{**input, "dimension": "X"}));
	Y = np.loadtxt(input["path"] + trackPattern.format(**{**input, "dimension": "Y"}));

	X = X*1.24;
	Y = Y*1.24;

	if X.ndim > 1:
		D = findMSD(X, Y);

		dX = np.diff(X, axis=1);
		dY = np.diff(Y, axis=1);
		dP = np.sqrt(dX**2 + dY**2);
		dP = dP[~np.isnan(dP)];
		dP = dP[dP < 50];

		jump = np.nanmean(dP);

		trackLength = np.mean(np.sum(1 - np.isnan(X), axis=1));

		initialCells = np.sum(1 - np.isnan(X[:, 0:3])) / 3;
		finalCells   = np.sum(1 - np.isnan(X[:, -3:])) / 3;		

		rows.append({
			"Celline"        : celline,
			"Density"        : density,
			"IsMotile"		 : celline2motile[celline],
			"MeanTrackLength": trackLength,
			"InitialCells"   : initialCells,
			"FinalCells"     : finalCells,
			"MeanJump"		 : jump
		});

		for n in range(dP[:].size):
			jumpRows.append({
				"Celline"        : celline,
				"Density"        : density,
				"IsMotile"		 : celline2motile[celline],
				"Jump"	     	 : dP[n]
			});



		'''
		ND, DD = findAttraction(X, Y);
		for n in range(len(ND)): 
			attractRows.append({
				"Celline"             : celline,
				"Density"             : density,
				"IsMotile"		      : celline2motile[celline],
				"NearestNeighbour"    : ND[n],
				"NearestNeighbourJump": DD[n]
			});
		'''
		'''
		ND, PS = distanceJump(X, Y);
		for n in range(len(ND)): 
			disJumpRows.append({
				"Celline"             : celline,
				"Density"             : density,
				"IsMotile"		      : celline2motile[celline],
				"NearestNeighbour"    : ND[n],
				"JumpDistance"        : PS[n]
			});
		'''
##
##
##
'''
df = pd.DataFrame(rows);
df["FinalInitialCellRatio"] = df["FinalCells"] / df["InitialCells"];


df.boxplot(by="Celline" , column="FinalInitialCellRatio");
df.boxplot(by="Density" , column="FinalInitialCellRatio");
df.boxplot(by="IsMotile", column="FinalInitialCellRatio");

plt.show();
'''

##
##
##
'''
import scipy.stats;

def densityPlot(df, by):
	motile2color = {0: "r", 1: "g", np.nan: "k"};

	for index, gdf in df.groupby(by=by):
		plt.figure();
		plt.title(index);
		#seaborn.distplot(gdf["Jump"], hist=False, kde=True, label=index, color=motile2color[celline2motile[index]])
		
		seaborn.distplot(gdf["Jump"], hist=False, kde=True);

		fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(gdf["Jump"], loc=0);

		x = np.arange(gdf["Jump"].min(), gdf["Jump"].max(), 0.1);
		y = scipy.stats.gamma.pdf(x=x, a=fit_alpha, scale=fit_beta, loc=fit_loc);

		plt.plot(x, y, "--");

		print(index, fit_alpha, fit_loc, fit_beta);


df = pd.DataFrame(jumpRows);

#plt.figure(); densityPlot(df, "Celline");
densityPlot(df, "Celline");

#plt.figure(); seaborn.violinplot(x="Celline" , y="Jump", data=df, inner="quartile", cut=0);
#plt.figure(); seaborn.violinplot(x="Density" , y="Jump", data=df, inner="quartile");
#plt.figure(); seaborn.violinplot(x="IsMotile", y="Jump", data=df, inner="quartile", cut=0);

plt.show();
'''

##
##
##
'''
def densityPlot(df, by):
	for index, gdf in df.groupby(by=by):
		seaborn.jointplot(x=gdf["NearestNeighbour"], y=gdf["NearestNeighbourJump"], kind="hex", color="k");

		plt.subplots_adjust(top=0.9);
		plt.suptitle(index);

df = pd.DataFrame(attractRows);


densityPlot(df, "Celline");
#densityPlot(df, "Density");
#densityPlot(df, "IsMotile");

plt.show();
'''

##
##
##
'''
def densityPlot(df, by):
	for index, gdf in df.groupby(by=by):
		#plt.figure();
		H, xs, ys = np.histogram2d(y=gdf["NearestNeighbour"], x=gdf["JumpDistance"], bins=(20, 10));

		x = (xs[1:] + xs[0:-1]) / 2;
		y = (ys[1:] + ys[0:-1]) / 2;

		M = x / np.sum(H, axis=1);

		H = H * np.matlib.repmat(M, H.shape[1], 1).T;

		H = np.median(H, axis=0);

		motile2color = {0: "r", 1: "g", np.nan: "k"};

		plt.plot(y, H, label=index, color=motile2color[celline2motile[index]]);
		plt.title(index);

df = pd.DataFrame(disJumpRows);


#densityPlot(df, "Celline");
#densityPlot(df, "Density");
#densityPlot(df, "IsMotile");

plt.show();
'''