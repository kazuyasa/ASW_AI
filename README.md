
## 画像分類モデル作成

- Docker起動
```bash
docker build -t kirara-docker .
docker run -it --rm -v ~/workspace/99.free/01.AI_KIRARA/01.aihub/app:/app kirara-docker
```
- 自動画像収集
```bash
# ChromeDriverのダウンロード必須
googleimagesdownload -ri -cd "chromedriver" -l 1000  -k "[女優名]"
```
- モデル作成
```bash
sh CreateModel.sh
```

## FlaskでWebアプリ化
- 仮想環境
```bash
source myvenv/bin/activate
```
- Flask起動
```bash
FLASK_APP=sample.py flask run
```