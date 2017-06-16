
scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	exp = np.exp(x)
	return exp/np.sum(exp,0)

print(softmax([10*s for s in scores])) #The probabilities get closer to 0 or 1
print(softmax([s/10 for s in scores])) #The probabilities get closer to the uniform distribution

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
