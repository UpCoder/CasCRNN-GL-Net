import matplotlib.pyplot as plt


def visualize_patch_size():
    xs = [3, 5, 7, 9]
    ys_dataset1 = [87.33, 83.57, 84.19, 85.11]
    ys_dataset2 = [79.19, 83.44, 84.09, 76.96]
    ys_mean = [83.26, 83.50, 84.14, 81.03]
    names = ['dataset1', 'dataset2', 'mean']
    plt.plot(xs, ys_dataset1, linewidth=1)
    plt.plot(xs, ys_dataset2, linewidth=1)
    plt.plot(xs, ys_mean, linewidth=1)
    plt.legend(names, loc='upper right')
    plt.xlabel('patch_size')
    plt.ylabel('accuracy')
    # plt.show()
    plt.savefig('./' + 'patch_size.svg')


if __name__ == '__main__':
    visualize_patch_size()