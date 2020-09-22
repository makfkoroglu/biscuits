import numpy as np
import pandas as pd
from sklearn.preprocessing  import LabelEncoder

veri= pd.read_csv("biscuits.csv") 
print (veri)
label_encoder=LabelEncoder().fit(veri.Shape)

labels=label_encoder.transform(veri.Shape)
classes=list(label_encoder.classes_)

x=veri.drop(["Shape"],axis=1)

y=labels

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x=sc.fit_transform(x)

from sklearn.model_selection import  train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


from tensorflow.keras.utils import to_categorical

y_train=to_categorical(y_train)

y_test=to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model=Sequential()
model.add(Dense(2,input_dim=1,activation="softmax"))
model.add(Dense(4,activation="relu"))
model.add(Dense(6,activation="tanh"))
model.add(Dense(8,activation="tanh"))
model.add(Dense(1,activation="softmax"))
model.compile(loss = "binary_crossentropy" , optimizer = "Adam" , metrics = ["accuracy"] )


from sklearn.model_selection import KFold
 
n_split=3
 
for train_index,test_index in KFold(n_split).split(x):
  x_train,x_test=x[train_index],x[test_index]
  y_train,y_test=y[train_index],y[test_index]
  

model.compile(loss="binary_crossentropy",optimizer="Adamax",metrics=["binary_accuracy"])

import keras
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
   
]
BATCH_SIZE = 400
EPOCHS = 10
history = model.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

print("Ortalama eğitim kaybı: ",np.mean(model.history.history["loss"]))
print("Ortalama eğitim başarımı: ",np.mean(model.history.history["binary_accuracy"]))
print("Ortalama doğrulama kaybı: ",np.mean(model.history.history["val_loss"]))
print("Ortalama doğrulama başarımı: ",np.mean(model.history.history["val_binary_accuracy"]))

import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(15,15))
ax1.plot(model.history.history['loss'],color='g',label="Eğitim kaybı")
ax1.plot(model.history.history['val_loss'],color='y',label="Doğrulama kaybı")
ax1.set_xticks(np.arange(20,100,20))
ax2.plot(model.history.history['binary_accuracy'],color='b',label="Eğitim başarımı")
ax2.plot(model.history.history['val_binary_accuracy'],color='r',label="Doğrulama başarımı")
ax2.set_xticks(np.arange(20,100,20))
plt.legend()
plt.show()