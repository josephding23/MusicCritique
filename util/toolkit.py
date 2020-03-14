import matplotlib.pyplot as plt

def plot_data(data):
    sample_data = data
    dataX = []
    dataY = []
    for time in range(64):
        for pitch in range(84):
            if sample_data[time][pitch] > 0.5:
                dataX.append(time)
                dataY.append(pitch)
    plt.scatter(x=dataX, y=dataY)
    plt.show()