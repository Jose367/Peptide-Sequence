import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
N = 3709
L = 130
Types = 27 #(alphabet + 1)
input_shape = (L, Types, 1)
num_classes = 47
batch_size = 128
epochs = 4
dictionary = ["aami | alpha-amylase inhibitor" ,"am | antiamnestic","ah | ACE inhibitor","ctox | celiac toxic","ab | antibacterial","at | antithrombotic,  im | immunomodulating","op | opioid","st | stimulating","is | immunostimulating","ne | neuropeptide  re | regulating","ac | anticancer","ao | antioxidative","con | contracting","lig | bacterial permease ligand","ai | anti inflammatory","nat | natriuretic","inh | inhibitor","che | chemotactic","acc | accelerating","he | haemolytic","an | anorectic","op1 | opioid agonist","op2 | opioid antagonist","rea | reacting","bin | binding","hyp | hypotensive","sti-dif | stimulating different activities","anbi | antibiotic","emb | embryotoxic","af | antifungal","avi | antiviral","vsc | vasoconstrictor","orp | Neuropeptide - orphan receptor GPR14 agonist","apr | activating ubiquitin-mediated proteolysis","fer | fertilisation-activating peptide","neur | neuroactive","bi | bitter taste peptide","um | umami","PKC | Protein Kinase C inhibitor","min | mineralising tissues and body fluids","mac | membrane -active peptide","hep_bin | heparin binding","dpp | dipeptidyl peptidase IV inhibitor","tox | toxic "]


df = pd.read_csv('Book2.csv', sep=',',header=None)
arr = df.values
print(arr.shape)
y = np.zeros((N))
x = np.zeros((N,L))
for i in range(N):
    y[i] = arr[i][0]
    for j in range(1,L+1):
        if arr[i][j] == arr[i][j] and arr[i][j] != '~': #not nan
            x[i][j-1] = ord(arr[i][j]) - 64
x = keras.utils.to_categorical(x)
x = x.reshape(N, L, Types, 1)
y = keras.utils.to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
	  validation_split=0.33,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

x = np.zeros(L)
arr = "FKMSJKS" #This is an example of a string to classify
for j in range(0,len(arr)):
    if arr[j] == arr[j] and arr[j] != '~': #not nan
        x[j] = ord(arr[j]) - 64
x = keras.utils.to_categorical(x, num_classes=Types)
x = x.reshape(1, L, Types, 1)
x = model.predict(x)
x = np.argmax(x)
print(dictionary[x])
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
