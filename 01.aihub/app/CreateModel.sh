#!/bin/bash

# 女優の画像を収集する
# googleimagesdownload -ri -cd "chromedriver" -l 1000  -k "紗倉まな"
# googleimagesdownload -ri -cd "chromedriver" -l 1000  -k "明日花キララ"
# googleimagesdownload -ri -cd "chromedriver" -l 300  -k "三上悠亜"
# googleimagesdownload -ri -cd "chromedriver" -l 200  -k "つぼみ AV女優"
# googleimagesdownload -ri -cd "chromedriver" -l 200  -k "上原亜衣" 

# 顔検出
python /app/src/detect_face.py

# テスト用画像の用意
python /app/src/devide_train_test.py

# 画像水増し
python /app/src/inflation.py

# 学習
python /app/src/learn.py

