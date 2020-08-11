import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def ReadCSV(fname: str):
    line_array = []
    with open(fname, "r") as csv:
        header = csv.readline().strip().split(", ")  # skip header
        for line in csv:
            line_array.append(line.strip().split(", "))
    line_array = [[float(val[:-2]) if val[-2:] == "ms" else float(val) for val in line] for line in line_array]
    return header, np.asarray(line_array)

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    header, line_array = ReadCSV(sys.argv[1])
    x_data = line_array[:, 0].reshape(-1, 4)
    log_x_data = np.log(x_data + 1)
    y_data = line_array[:, 1].reshape(-1, 4)
    arranged_data = line_array[:, 2:].reshape(len(header) - 2, -1, 4)
    fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
    gridspec = fig.add_gridspec(2, 4)
    colors = ['purple', 'red', 'orange', 'green', 'blue']
    axes = [fig.add_subplot(gridspec[:, :2]),
            fig.add_subplot(gridspec[0, 2]),
            fig.add_subplot(gridspec[0, 3]),
            fig.add_subplot(gridspec[1, 2]),
            fig.add_subplot(gridspec[1, 3])]
    z_limit = [np.min(arranged_data[:, :, 0]), np.max(arranged_data[:, :, 0])]
    log_z_limit = [-0.2, np.log(z_limit[1] + 1)]
    # for i in range(2, len(header)):
    #     current_data = np.log(line_array[:, i].reshape(-1, 4))
    #     axes[i - 2].plot_surface(x_data, y_data, current_data, color=colors[i - 2])
    #     axes[i - 2].set_xlim3d(np.min(x_data), np.max(x_data))
    #     axes[i - 2].set_xlabel("(log) " + header[0])
    #     axes[i - 2].set_ylim3d(np.min(y_data), np.max(y_data))
    #     axes[i - 2].set_ylabel(header[1])
    #     axes[i - 2].set_zlim3d(*z_limit)
    #     axes[i - 2].set_zlabel("(log) Time in ms")
    #     axes[i - 2].view_init(20, -110)
    #     axes[i - 2].title.set_text(header[i][:-10] if header[i][-10:] == " Time (ms)" else header[i])
    x_tick_labels = np.unique(x_data).astype(np.int)[::2]
    x_ticks = np.log(x_tick_labels + 1)
    z_ticks = np.linspace(0, np.ceil(log_z_limit[1]), 8)
    z_tick_labels = (np.exp(z_ticks) - 1).astype(np.int)
    for i in range(2, len(header)):
        current_data = np.log(line_array[:, i].reshape(-1, 4) + 1)
        axes[i - 2].plot(log_x_data[:, 0], current_data[:, 0], color=colors[i - 2])
        axes[i - 2].set_xlabel(header[0], fontsize=12)
        axes[i - 2].set_xticks(x_ticks)
        axes[i - 2].set_xticklabels(x_tick_labels)
        axes[i - 2].set_ylabel("Time in ms", fontsize=12)
        axes[i - 2].set_ylim(*log_z_limit)
        axes[i - 2].set_yticks(z_ticks)
        axes[i - 2].set_yticklabels(z_tick_labels)
        axes[i - 2].set_title(header[i][:-10] if header[i][-10:] == " Time (ms)" else header[i], fontsize=12)
    plt.show()