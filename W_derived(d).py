import matplotlib.pyplot as plt
import numpy as np

n = 100
w, I, E0, c = 1, 0.5, 1, 0.5

def W_derived(d):
    total = 0.0
    for k in range(-n, n + 1):
        ak = ((k - 0.5) * d) ** 2
        total += (c * c - ak) / (ak + c * c) ** 2
    
    return total

x = np.linspace(0, 5, 200)
y = W_derived(x)

plt.plot(x, y)
plt.plot((0, 5), (0, 0), linestyle='--')
plt.xlim(0, 5)
plt.ylim(-2, 2)
plt.xticks(np.linspace(0, 5, 6))
plt.yticks(np.linspace(-2, 2, 5))
plt.show()