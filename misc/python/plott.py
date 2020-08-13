import matplotlib.pyplot as plt

def pplt():
    data1 = [13189, 35136, 16827, 25911, 68034]
    data2 = [68893, 62697, 72811, 106038, 69770]
    x = range(5)

    plt.plot(x, data1, marker='o', label='redis')
    plt.plot(x, data2, marker='*', label='hbase')

    plt.xticks(range(5), ['8(7+1)', '10(7+3)', '10(9+1)', '15(8+7)', '15(7+8)'])

    plt.xlabel('Scenario')
    plt.ylabel('time (ms)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    pplt()