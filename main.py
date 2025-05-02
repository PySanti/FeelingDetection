from tensorflow.keras import datasets
from utils.preprocess_data import preprocess_data

(X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data()
X_train, X_test = preprocess_data(X_train, X_test, 500)

