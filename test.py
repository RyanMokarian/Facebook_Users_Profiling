import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

x = np.random.randn(1000, 3)
# x = [4,6,7,7,9,10,10,11]


colors = ['red', 'tan', 'lime']
plt.hist(x, 4, density=True, histtype='bar', color=colors, label=colors)
plt.show().legend(prop={'size': 10})
plt.show().set_title('bars with legend')
plt.show()