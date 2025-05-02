from tensorflow.keras import datasets
from utils.preprocess_data import preprocess_data
from sklearn.model_selection import train_test_split

(X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data()
X_train, X_test = preprocess_data(X_train, X_test, 500)

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, random_state=42, stratify=Y_test, test_size=0.4)

print("shape de train")
print(X_train.shape)
print(Y_train.shape)

print("shape de val")
print(X_val.shape)
print(Y_val.shape)


print("shape de test")
print(X_test.shape)
print(Y_test.shape)
