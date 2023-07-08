import matplotlib.pyplot as plt
import numpy as np

# Crea la primera figura
fig1 = plt.figure()
x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.title('Gráfico 1')

# Crea la segunda figura
fig2 = plt.figure()
x = np.arange(0, 2*np.pi, 0.1)
y = np.cos(x)
plt.plot(x, y)
plt.title('Gráfico 2')

# Muestra ambas figuras
plt.show()
