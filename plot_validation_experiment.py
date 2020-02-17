import numpy as np;
import pandas as pd;
from matplotlib import pyplot as plt;

###########
## Input ##
###########
dpi = 92;

basePath = "/media/emiro593/AdherentCells/";


df = pd.read_csv(basePath + "validation_abc_posterior.csv", sep="\t");
df["GroupLabel"] = df.apply(lambda row: "{}_{}_{}_{}{}".format(row["Celline"], row["Plate"], row["Density"], row["Row"], row["Col"]), axis=1);


def plot(df):
	plt.figure();
	plt.hist(df["Proliferation"], bins=20);
	plt.title(df.index.values);


for line, sdf in df.groupby(by="Celline"):
	print(line);

	plt.figure();
	plt.suptitle(line);

	N = len(sdf["GroupLabel"].unique());
	S = np.ceil(np.sqrt(N));

	for i, (label, wdf) in enumerate(sdf.groupby(by="GroupLabel")):

		print(line, i, label);

		plt.subplot(S, S, i + 1);
		plt.hist((10**wdf["Proliferation"]) * 24 * 3600, bins=20);

plt.show();