import os
import cv2
import glob
from scipy import ndimage
'''
学習量データの水増し処理を実行する
'''
# 学習対象の女優一覧
names = ["asuka","mana","mikami","tsubomi","uehara"]
os.makedirs("/app/resources/train", exist_ok=True)
#  HACK:
for name in names:
    in_dir = "/app/resources/face/"+name+"/*"
    out_dir = "/app/resources/train/"+name
    os.makedirs(out_dir, exist_ok=True)
    in_jpg=glob.glob(in_dir)
    img_file_name_list=os.listdir("/app/resources/face/"+name+"/")
    for i in range(len(in_jpg)):
        img = cv2.imread(str(in_jpg[i]))
        # 回転
        for ang in [-10,0,10]:
            img_rot = ndimage.rotate(img,ang)
            img_rot = cv2.resize(img_rot,(64,64))
            fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+".jpg")
            cv2.imwrite(str(fileName),img_rot)
            # 閾値
            img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
            fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+"thr.jpg")
            cv2.imwrite(str(fileName),img_thr)
            # ぼかし
            img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
            fileName=os.path.join(out_dir,str(i)+"_"+str(ang)+"filter.jpg")
            cv2.imwrite(str(fileName),img_filter)