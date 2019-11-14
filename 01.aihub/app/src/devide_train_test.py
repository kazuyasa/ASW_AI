import shutil
import random
import glob
import os
'''
2割の画像をテストデータ用のディレクトリに移動する
'''

# 学習対象の女優
names = ["asuka","mana","mikami","tsubomi","uehara"]

# テストデータ用のディレクトリ作成
os.makedirs("/app/resources/test", exist_ok=True)

# HACK: 
for name in names:
    in_dir = "/app/resources/face/"+name+"/*"
    in_jpg=glob.glob(in_dir)
    img_file_name_list=os.listdir("/app/resources/face/"+name+"/")
    #　ランダムな2割の画像をテスト用ディテクトリに移動させる
    random.shuffle(in_jpg)
    os.makedirs('/app/resources/test/' + name, exist_ok=True)
    for t in range(len(in_jpg)//5):
        shutil.move(str(in_jpg[t]), "/app/resources/test/"+name)