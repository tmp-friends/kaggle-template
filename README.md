# KaggleTemplate

https://github.com/ktakita1011/my_kaggle_docker

## What is this

kaggle docker can use GPU

worked on docker version 24.x.x higher

how to build and run
```bash
$ cd kaggle-template
$ docker compose up -d
```

## Acess jupyter notebook
acess here http://127.0.0.1:8888
default jupyter notebook password is "kaggle"
if u want to change password, look run.sh

## Attach to a running container
In Visual Code, using attach to a running container.
https://code.visualstudio.com/docs/devcontainers/attach-container#_attach-to-a-docker-container

## KaggleAPI

https://github.com/Kaggle/kaggle-api

**以下の作業は、ホストPC上で行う**

### （初回のみ）KaggleAPIライブラリのSetup

- KaggleAPIのインストール

```
$ pip install kaggle
```

- KaggleAPIトークンを書き込む

```
$ vim ~/.kaggle/kaggle.json
```

### APIの操作

https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#commands

