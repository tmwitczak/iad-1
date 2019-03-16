import matplotlib.pyplot as plt

# ////////////////////////////////////////// To delete
from sklearn.datasets import load_sample_image
from sklearn.cluster import MiniBatchKMeans


# //////////////////////////////////////////


class KMeansAlgorithm:
    # ////////////////////////////////////////// To delete
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    def imageCompressionFromTutorial(self):
        china = load_sample_image("china.jpg")
        data = china / 255.0  # use 0...1 scale
        data = data.reshape(427 * 640, 3)
        kmeans = MiniBatchKMeans(16)
        kmeans.fit(data)
        new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
        china_recolored = new_colors.reshape(china.shape)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(wspace=0.05)
        ax[0].imshow(china)
        ax[0].set_title('Original Image', size=16)
        ax[1].imshow(china_recolored)
        ax[1].set_title('16-color Image', size=16)
        plt.show()
    # //////////////////////////////////////////
    # def __init__(self):

