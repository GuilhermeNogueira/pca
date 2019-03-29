from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA

if __name__ == '__main__':

    # Load MNIST dataset
    mnist = fetch_mldata("MNIST original")
    X, y = mnist.data / 255.0, mnist.target

    # Create subset and reduce to first 50 dimensions
    indices = arange(X.shape[0])
    random.shuffle(indices)
    n_train_samples = 5000
    X_pca = PCA(n_components=50).fit_transform(X)
    X_train = X_pca[indices[:n_train_samples]]
    y_train = y[indices[:n_train_samples]]
