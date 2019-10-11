import tensorflow as tf
config = tf.ConfigProto()
sess = tf.Session(config=config)

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def get_session(gpu_fraction=0.9):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\git\Machine-Learning-A-Z-Udemy\data_files\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# ------ Part-1: Data preprocessing ----------

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# ------- Part-2: Build the ANN --------

# import keras library and packages
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
layer_info = Dense(activation='relu',  # É o uso da função rectifier
				   input_dim=11,  # Número de variáveis de entrada
				   kernel_initializer='uniform',  # Inicializa os pesos de acordo com a distribuição uniforme
				   units=6)  # units é o numero de neurons da hidden layer.
# Uma regra básica é usar "(Variaveis de entrada + Variaveis de saida) / 2
classifier.add(layer_info)

# Adding second hidden layer, não precisa do parametro input_dim pois só no primeiro precisa.
layer_info = Dense(activation='relu', kernel_initializer='uniform', units=6)
classifier.add(layer_info)

# Adding output layer
layer_info = Dense(activation='sigmoid',  # Porque é a camada de caída, se tivesse 3 saidas ou mais seria softmax
				   kernel_initializer='uniform',
				   units=1)
classifier.add(layer_info)

# Compiling the ANN
classifier.compile(optimizer='adam',  # É um stochastic gradient descend usado para descobrir os pesos.
				   loss='binary_crossentropy',  # Para calcular o cost function, essa função é comum em sigmoids.
				   metrics=['accuracy'])  # A metrica para verificar a performance do modelo.

import time
start = time.time()
# Fitting the ANN to the training set
classifier.fit(X_train, y_train,
			   batch_size=100,  # Número de observações antes que se atualize os weights
			   epochs=50)
print (time.time()-start)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

