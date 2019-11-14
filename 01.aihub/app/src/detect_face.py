import glob
import os
import cv2
'''
dataディレクトリの画像から顔検出してfaceディレクトリに保存する
'''
# 収集対象の女優
names = ["asuka","mana","mikami","tsubomi","uehara"]

# 顔検出画像の保存用ディレクトリ作成
out_dir = "/app/resources/face/"
os.makedirs(out_dir, exist_ok=True)

for i in range(len(names)):
    # 元画像の顔部分を正方形で囲み、64×64pにリサイズして保存する
    in_dir = "/app/resources/data/"+names[i]+"/*.jpg"
    in_jpg = glob.glob(in_dir)
    os.makedirs(out_dir + names[i], exist_ok=True)
    for num in range(len(in_jpg)):
        image=cv2.imread(str(in_jpg[num]))
        if image is None:
            print("Not open:",num)
            continue

        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier('/app/resources/model/haarcascade_frontalface_default.xml')
        # 顔認識の実行
        face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
        #顔が１つ以上検出された時
        if len(face_list) > 0:
            for rect in face_list:
                x,y,width,height=rect
                image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image.shape[0]<64:
                    continue
                image = cv2.resize(image,(64,64))
                #保存
                fileName=os.path.join(out_dir+"/"+names[i],str(num)+".jpg")
                cv2.imwrite(str(fileName),image)
                print(str(num)+".jpgを保存しました.")
        #顔が検出されなかった時
        else:
            print("no face")
            continue
        print(image.shape)