import matplotlib.pyplot as plt
import numpy as np


info_labels = ['Cycle Consistency Loss', 'Generator Loss', 'Discriminator Loss']


loss_dict = {
    'mc': {
        'old': [0.110, 0.654, 0.468],
        'new': [0.121, 0.657, 0.453]
    },
    'pc': {
        'old': [0.087, 0.780, 0.453],
        'new': [0.097, 0.772, 0.432]
    },
    'rj': {
        'old': [0.049, 0.657, 0.465],
        'new': [0.081, 0.617, 0.467]
    }
}

group = 'mc'

size = 3
x = np.arange(size)
y1 = loss_dict[group]['old']
y2 = loss_dict[group]['new']

total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, y1,  width=width, label='a')
plt.bar(x + width, y2, width=width, label='b')
plt.legend()
plt.show()
