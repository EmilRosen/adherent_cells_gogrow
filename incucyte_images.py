import os;
import parse;

from tracking import utils;

def load(basePath, pattern, millenia="20"):
	monthdir = os.listdir(basePath);

	files = [];
	for ym in monthdir:
		daydirs = os.listdir(basePath + "/" + ym);

		for d in daydirs:
			timedirs = os.listdir(basePath + "/" + ym + "/" + d);

			for t in timedirs:
				platedirs = os.listdir(basePath + "/" + ym + "/" + d + "/" + t);

				for p in platedirs:
					if p.isdigit() == False:
						continue;

					platePath = basePath + "/" + ym + "/" + d + "/" + t + "/" + p;

					year   = int(millenia + ym[0:2]);
					month  = int(ym[2:4]);
					day    = int(d);
					hour   = int(t[0:2]);
					minute = int(t[2:4]);

					plate = int(p);

					#print(year, month, day, hour, minute, plate);

					inputs = utils.getInputFiles(pattern, platePath);

					for i in range(len(inputs)):
						inputs[i]["year"]   = year;
						inputs[i]["month"]  = month;
						inputs[i]["day"]    = day;
						inputs[i]["hour"]   = hour;
						inputs[i]["minute"] = minute;

						inputs[i]["plate"] = plate;

					files = files + inputs;

	return files;