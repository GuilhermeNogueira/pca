from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if __name__ == '__main__':

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Using PCA to reduce iris dataset dimensions
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure(2, figsize=(8, 6))
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title('PCA of IRIS dataset')

    plt.show()
