# Testing Different Segementation Algorithims
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt
from skimage import color, segmentation
from skimage import graph as sk_graph


def n_cut(image, segments, visualize=False):
    """
    Simple N-Cut Segmentation
    """
    labels1 = segmentation.slic(
        image, compactness=80, n_segments=segments, start_label=1
    )
    out1 = color.label2rgb(labels1, image, kind="avg", bg_label=0)

    g = sk_graph.rag_mean_color(image, labels1)
    labels2 = sk_graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, image, kind="avg", bg_label=0)

    if visualize:
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

        ax[0].imshow(out1)
        ax[1].imshow(out2)

        for a in ax:
            a.axis("off")

        plt.tight_layout()
        plt.show()


def k_means(image, clusters, visualize=False):
    """
    Simple K-Means Segmentation
    """
    kmeans = KMeans(n_clusters=clusters, n_init=10, verbose=0).fit(image.reshape(-1, 3))
    kmeans_result = color.label2rgb(
        kmeans.labels_.reshape(image.shape[0], -1), image, kind="avg"
    )
    if visualize:
        plt.subplot(1, 3, 3)
        plt.title("Kmeans")
        plt.xticks([]), plt.yticks([])
        plt.imshow(kmeans_result)
        plt.show()


def mean_shift(image, bandwith, visualize=False):
    """
    Simple Mean-Shift Segmentation
    """
    # Flatten the Image
    flat_image = image.reshape((-1, 3))
    flat_image = np.float32(flat_image)

    # Mean-Shift Segmentation
    bandwidth_estimation = estimate_bandwidth(flat_image, quantile=0.06, n_samples=3000)
    ms = MeanShift(bandwidth=bandwith, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled = ms.labels_

    # Count and Average Colors
    segments = np.unique(labeled)
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total / count
    avg = np.uint8(avg)

    # Create Result Image
    res = avg[labeled]
    result = res.reshape((image.shape))

    if visualize:
        plt.title("Meanshift")
        plt.xticks([]), plt.yticks([])
        plt.imshow(result)
        plt.show()
    return len(segments)
