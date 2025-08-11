# Kaggle Competition Template

Kaggle コンペティション用の汎用MLパイプラインテンプレートです。表形式、画像、音声、テキストデータなど、様々なタイプのコンペティションに対応できます。

## 📋 プロジェクト概要

このテンプレートは、どのようなKaggleコンペティションにも適用できる標準化されたMLパイプラインを提供します。Hydraによる実験管理、uvによる最新のPythonパッケージ管理、様々なMLアプローチへの包括的なサポートが含まれています。

**🔧 詳細な設定と高度な使用方法については、[CLAUDE.md](CLAUDE.md) を参照してください**

## 🚀 クイックスタート

### 1. 環境構築

[uv](https://docs.astral.sh/uv/)（高速Pythonパッケージマネージャー）のインストール：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

仮想環境の作成と依存関係のインストール：
```bash
uv sync
```

### 2. データ準備

コンペティションデータのダウンロード：
```bash
cd input/
kaggle competitions download -c {competition_name}
unzip {competition_name}.zip -d ./{competition_name}
rm -f {competition_name}.zip
```

### 3. モデル訓練

基本的な訓練の実行：
```bash
uv run python src/01-train.py fold=0
```

クロスバリデーションの実行：
```bash
uv run python src/01-train.py --multirun fold=0,1,2,3,4
```

### 4. 推論

予測の生成：
```bash
uv run python src/02-infer.py
```

## 🏗️ プロジェクト構成

```
.
├── README.md               # このファイル
├── CLAUDE.md              # 詳細な設定ガイド
├── pyproject.toml         # 依存関係とプロジェクト設定
├── uv.lock               # ロックファイル
├── input/                # コンペティションデータ
├── src/                  # ソースコード
│   ├── conf/            # Hydra設定ファイル
│   ├── datasets/        # データセット定義
│   ├── models/          # モデル定義
│   ├── utils/           # ユーティリティ関数
│   └── *.py            # 訓練・推論スクリプト
├── output/              # 実験結果
├── multirun/            # Hydraマルチラン結果
└── notebook/            # Jupyter notebooks
```

## 🛠️ 技術スタック

- **パッケージ管理**: uv（高速Pythonパッケージマネージャー）
- **実験管理**: Hydra（設定管理フレームワーク）
- **MLライブラリ**: PyTorch、scikit-learn、LightGBM、XGBoost、CatBoost
- **コンピュータビジョン**: timm、torchvision、albumentations、OpenCV
- **データ処理**: pandas、polars、numpy
- **可視化**: matplotlib、seaborn、japanize-matplotlib
- **テスト**: pytest

## 📊 対応するコンペティションタイプ

### 🗂️ 表形式データ
- 勾配ブースティング（LightGBM、XGBoost、CatBoost）
- 特徴量エンジニアリングパイプライン
- クロスバリデーション戦略

### 🖼️ 画像データ
- CNNアーキテクチャ（ResNet、EfficientNet、Vision Transformer）
- albumentationsによるデータ拡張
- timmの事前学習済みモデル

### 🎵 音声データ
- メルスペクトログラム変換
- 音声データ拡張
- CNN/RNNの組み合わせ

### 📝 テキストデータ
- 事前学習言語モデル（BERT、RoBERTa）
- トークン化とベクトル化
- ファインチューニング戦略

## ⚙️ 設定管理

このテンプレートでは、柔軟な実験管理のためにHydraを使用しています：

```bash
# 学習率を変更
uv run python src/01-train.py lr=1e-3

# バッチサイズを変更
uv run python src/01-train.py train_batch_size=64

# 異なるモデルを使用
uv run python src/01-train.py model=ResNet50

# 複数のパラメータを変更
uv run python src/01-train.py lr=1e-3 num_epochs=30 fold=0
```

## 📦 パッケージ管理

新しいパッケージの追加：
```bash
uv add "package-name==version"
```

パッケージの削除：
```bash
uv remove package-name
```

ロックファイルの更新：
```bash
uv lock
```

## 🔄 Kaggle提出ワークフロー

1. **ローカルでモデルを訓練**
2. **モデルデータセットを作成**（`output/upload/`）
3. **コードデータセットを作成**（`src/`）
4. **Kaggle Datasetsにアップロード**
5. **Kaggle Notebookで推論を実行**

詳細なKaggle提出プロセスについては、[CLAUDE.md](CLAUDE.md)を参照してください。

## 📚 ドキュメント

- **[CLAUDE.md](CLAUDE.md)** - Claude Codeユーザー向け包括的ガイド
- **[pyproject.toml](pyproject.toml)** - 依存関係とプロジェクト設定

## 🤝 貢献

このテンプレートは柔軟で拡張可能になるように設計されています。特定のコンペティションのニーズに合わせて自由に調整してください。

### 参考

- uv: https://docs.astral.sh/uv/
- uvの使用例: https://zenn.dev/turing_motors/articles/594fbef42a36ee
- tubo213-san kaggle環境: https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states
- Docker kaggle環境: https://github.com/ktakita1011/my_kaggle_docker
- kaggle api: https://github.com/Kaggle/kaggle-api
