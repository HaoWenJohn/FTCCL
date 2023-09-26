import numpy as np
import matplotlib.pyplot as plt

def draw(title,data,save_path=None):
    # Set up the plot
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="YlGnBu",vmin=0, vmax=1)

    # Add labels, title, and colorbar
    ax.set_xticks(np.arange(len(title)))
    ax.set_yticks(np.arange(len(title)))
    ax.set_xticklabels(title)
    ax.set_yticklabels(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im)

    # Add text annotations
    thresh = data.max() / 2.
    for i in range(len(title)):
        for j in range(len(title)):
            ax.text(j, i, format("%.2f" % data[i, j] ),
                    ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black")
    if save_path is not None:
    # Show the plot
        plt.savefig(save_path)
    else:plt.show()

