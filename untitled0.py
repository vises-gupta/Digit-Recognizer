import pandas as pd
import numpy as np

df=pd.read_csv("train.csv")
dft=pd.read_csv("test.csv")


X_train=df.iloc[:,1:].values
y_train=df.iloc[:,0].values

X_test=dft.iloc[:,:].values



from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
from keras.utils import np_utils
y_train= np_utils.to_categorical(encoded_Y)

import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising ANN
classifier=Sequential()

#adding input and hidden layers
classifier.add(Dense(393, kernel_initializer='uniform',activation='relu',input_dim=784))
#adding another hidden layer
classifier.add(Dense(393, kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(197, kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(99, kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(50, kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(26, kernel_initializer='uniform',activation='relu'))



#adding output layer
classifier.add(Dense(10, kernel_initializer='uniform',activation='softmax'))


#compile
classifier.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])

#training
classifier.fit(X_train,y_train,batch_size=32,epochs=30)

y_pred=classifier.predict(X_test)

pred=[]

for i in range(0,28000):
    li=list(y_pred[i])
    val=max(li)
    pred.append(li.index(val))

 #to save predicted value in dataframe .csv file

DF1 = pd.DataFrame(list(pred))
DF1.columns=['Label']
DF1.to_csv("ANN1.csv")


pred=np.array(pred)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,pred)
print(accuracy_score(y_test,pred))
