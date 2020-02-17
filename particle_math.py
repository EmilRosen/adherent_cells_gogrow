import numpy as np;
from scipy.spatial.distance import cdist;

def particlesInRectangle(P, rect):
	leftEdgeMap   = P[:, 0] >= rect[0];
	rightEdgeMap  = P[:, 0] <= rect[2];
	bottomEdgeMap = P[:, 1] >= rect[1];
	topEdgeMap    = P[:, 1] <= rect[3];

	interior = leftEdgeMap * rightEdgeMap * bottomEdgeMap * topEdgeMap;

	return np.sum(interior);

def pairCorrelation(P, rect, rMax, dr):
	edges = np.arange(0, rMax, dr);
	radii = (edges[:-1] + edges[1:]) / 2;

	if P.size == 0:
		return np.zeros(shape=radii.shape), radii;

	leftEdgeMap   = P[:, 0] >= rect[0] + rMax;
	rightEdgeMap  = P[:, 0] <= rect[2] - rMax;
	bottomEdgeMap = P[:, 1] >= rect[1] + rMax;
	topEdgeMap    = P[:, 1] <= rect[3] - rMax;

	interior = leftEdgeMap * rightEdgeMap * bottomEdgeMap * topEdgeMap;	# Find interior particles to avoid edge effects

	density = particlesInRectangle(P, rect) / ((rect[2] - rect[0]) * (rect[3] - rect[1]));	# Particles per unit

	D = cdist(P, P, metric='euclidean');	# Compute distance matrix
	
	np.fill_diagonal(D, np.nan);
	
	D = D[interior, :];						# Only use the interior particles as reference particles
	D = D[~np.isnan(D)];

	areas = np.pi * np.diff(edges**2);		# Area of donut shaped strips

	results, VOID = np.histogram(D.flatten(), bins=edges, normed=False);	# Count number of particles in each donut-shaped bin

	results = results / (np.sum(interior) * areas * density);

	return results, radii;



