#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, request, redirect, url_for, jsonify, send_from_directory
import numpy as np
import cv2
from keras.models import  load_model
import sys
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename


# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './predict'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allwed_file(filename):
    # 拡張子の確認
    # OK=１ NG=0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# GETでアクセスされた場合には、画像UPLOAD用のフォーム画面に遷移
# TODO: 外部HTMLに変更する
@app.route('/', methods=['GET'])
def return_form():
    return '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>
                UPLOAD
            </title>
        </head>
        <body>
            <h1>
                UPLOAD
            </h1>
            <form method = post enctype = multipart/form-data>
            <p><input type=file name = file>
            <input type = submit value = Upload>
            </form>
        </body>
    '''

# ファイルを受け取る方法の指定
@app.route('/', methods=['POST'])
def predict_image():
    try:
        # ファイルがなかった場合の処理
        # HACK:
        if 'file' not in request.files:
            raise Exception
            # result='This video has been deleted'
            # return jsonify({'result': str(result)}) 
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        # HACK:
        if file.filename == '':
            raise Exception
            # result='This video has been deleted'
            # return jsonify({'result': str(result)}) 
        # ファイルのチェック
        if file and allwed_file(file.filename):
            # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(file.filename)
            # ファイルの保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # 予測
            result=predict(filename)
            # 予測結果を返す
            return jsonify({'result': str(result)})     
    except:
        return jsonify({'result': 'This Video Has Been Deleted'})
    


# ユーザーがPOSTした画像から顔を切り出す
def detect_face(image):
    print("DETECT FACE")
    print(image.shape)
    #　顔抽出
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 顔検出用の機械学習モデルの読み込み
    cascade = cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    # 顔が検出された時
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
            img = cv2.resize(image,(64,64))
            img=np.expand_dims(img,axis=0)
            # name = detect_who(img)
            # cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
            return img
    # 顔が検出されなかった時
    else:
        result='This video has been deleted (no face)'
        return jsonify({'result': str(result)}) 
    # return image
    # return name

def predict_who(img):
    print("PREDICT WHO")
    # 予測
    name = ""
    model = load_model('./model/asw_ai.h5')
    nameNumLabel=np.argmax(model.predict(img))
    if nameNumLabel== 0: 
        name="Asuka Kirara"
    elif nameNumLabel==1:
        name="Mana Sakura"
    elif nameNumLabel==2:
        name="Yua Mikami"
    elif nameNumLabel==3:
        name="tsubomi"
    elif nameNumLabel==4:
        name="Ai uehara"
    return name

# 画像の予測ロジック
def predict(img_jpg):
    print(" START PREDICTION ")
    # 予測対象の画像を読み込み
    image=cv2.imread("./predict/"+img_jpg)
    if image is None:
        result = 'This video has been deleted (no image)'
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    # 顔検出
    face_image = detect_face(image)
    # 予測
    result = predict_who(face_image)
    print("result is {}".format(result))
    return (result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)