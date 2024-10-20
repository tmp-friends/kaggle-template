## KaggleTemplate

### Build Environment

#### 1. install [uv](https://docs.astral.sh/uv/)

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Create virtual environment

`pyproject.toml`, `uv.lock`の内容を元に仮想環境を構築

```
$ uv sync
```

#### 3. Activate virtual environment

```
$ . .venv/bin/activate
```

#### 4. Add/Remove package

e.g. numpy v1.26.4

```
$ uv add "numpy==1.26.4"
```

```
$ uv remove numpy
```

`uv.lock`の更新

```
$ uv lock
```

### Prepare Data

inputフォルダにデータをDownload

```
$ cd input/
$ kaggle competitions download -c {competition_name}
$ unzip {competition_name}.zip -d ./{competition_name}
$ rm -f {competition_name}.zip
```

### Train model

- Hydra

### Infer

#### Upload model dataset

- `output/upload`フォルダに学習したモデルをcp

- Metadata json fileの生成

```
$ kaggle datasets init -p output/upload
```

- `dataset-metadata.json`を編集

```json
{
  "title": "${competition_name}-models",
  "id": "komekami/${competition_name}-models",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
```

- Kaggle Datasetの作成

```
$ kaggle datasets create -p output/upload --dir-mode zip
```

- Kaggle Datasetの更新

```
$ kaggle datasets version -p output/upload -m 'hoge' --dir-mode zip
```


#### Upload repo dataset

- Metadata json fileの生成

```
$ kaggle datasets init -p src
```

- `dataset-metadata.json`を編集

```json
{
  "title": "${competition_name}-repo",
  "id": "komekami/${competition_name}-repo",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
```

- Kaggle Datasetの作成

```
$ kaggle datasets create -p src --dir-mode zip
```

- Kaggle Datasetの更新

```
$ kaggle datasets version -p src -m 'hoge' --dir-mode zip
```

#### Kaggle notebook上でInfer実行



### 参考

- uv: https://docs.astral.sh/uv/
- uvの使用例: https://zenn.dev/turing_motors/articles/594fbef42a36ee
- tubo213-san kaggle環境: https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states
- Docker kaggle環境: https://github.com/ktakita1011/my_kaggle_docker
- kaggle api: https://github.com/Kaggle/kaggle-api
