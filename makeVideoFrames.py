import os
import numpy as np
import pickle
import plot


# Load results file.
with open('results1.bin', 'rb') as pickleFile:
	results = pickle.load(pickleFile)
print("Results loaded.")

exportPath = './videos/frames/'

# Sort wrt episode reward.
episodeRewards = [np.sum(r['reward']) for r in results]
resultIndices = np.argsort(episodeRewards)

# Pick every 100th.
resultIndices = list(reversed(resultIndices))[0:10]

# Walk through picked episodes.
for iE, e in enumerate(resultIndices):
	episodePath = 'video{:04}_episode{:04}/'.format(iE, e)
	fullPath = exportPath + episodePath
	# Walk through all frames and plot.
	if not os.path.exists(fullPath):
	    os.makedirs(fullPath)
	for i in range(len(results[e]['time'])):
		print(i)
		# Create image
		plot.plot(results[e], fancy=False, plotDims=[22,48], runtime=10, framesMax=i, filepath=fullPath+'frame{:04}.png'.format(i))

