import os
import cv2
import numpy as np
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

# 予測対象の女優一覧
name = ["asuka","mana","mikami","tsubomi","uehara"]

# 教師データのラベル付け
X_train = [] 
Y_train = [] 
for i in range(len(name)):
    img_file_name_list=os.listdir("/app/resources/train/"+name[i])
    print(len(img_file_name_list))
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("/app/resources/train/"+name[i]+"/",img_file_name_list[j])
        print(img_file_name_list[j])
        img = cv2.imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)

# テストデータのラベル付け
X_test = [] # 画像データ読み込み
Y_test = [] # ラベル
for i in range(len(name)):
    img_file_name_list=os.listdir("/app/resources/test/"+name[i])
    print(len(img_file_name_list))
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("/app/resources/test/"+name[i]+"/",img_file_name_list[j])
        img = cv2.imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_test.append(img)
        # ラベルは整数値
        Y_test.append(i)
X_train=np.array(X_train)
X_test=np.array(X_test)

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# モデルの定義
model = Sequential()
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(5))
model.add(Activation('softmax'))

# コンパイル
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 学習
history = model.fit(X_train, y_train, batch_size=32, 
                    epochs=10, verbose=1, validation_data=(X_test, y_test))

# 汎化制度の評価・表示
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))


#　モデルを保存
model.save("asw_ai.h5")
