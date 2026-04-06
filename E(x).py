import matplotlib.pyplot as plt
import numpy as np

n = 100
d, w, I = 2, 1, 0.5

def g(x, k):
    return I / ((k * d - x) ** 2 + w * w / 4)

def f(x):
    s = 0
    s += g(x, 0)
    for i in range(1, n + 1):
        s += g(x, i) + g(x, -i)
    return s

x = np.linspace(-5, 5, 200)
y = f(x)

plt.plot(x, y)
plt.xlim(-5, 5)
plt.ylim(0, 3)
plt.xticks(np.linspace(-5, 5, 11))
plt.yticks(np.linspace(0, 3, 7))
plt.show()