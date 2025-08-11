# Kaggle Competition Development Environment

Claude Codeが効率的にKaggleコンペティションプロジェクトで作業するための汎用的な環境情報とガイドライン。

## プロジェクト概要

このプロジェクトはKaggleコンペティション用のMLパイプラインテンプレートです。様々なタイプのコンペティション（表形式、画像、音声、テキストなど）に対応できる汎用的な構成となっています。

詳細は、以下のファイルに説明があるので、必要な場面ではそちらを参照してください。
- コンペティションの概要: `.claude/overview.md`
- データセットの説明: `.claude/datasets.md`

## 技術スタック

### パッケージ管理・仮想環境
- **uv**: 高速Pythonパッケージマネージャー
- **pyproject.toml**: 依存関係とプロジェクト設定の管理

### 実験・設定管理
- **Hydra**: 設定管理フレームワーク
  - YAML設定ファイルによる柔軟な実験設定
  - マルチランサポート
  - 階層的設定の合成

### 機械学習
- **PyTorch**: ディープラーニングフレームワーク
- **scikit-learn**: 機械学習ライブラリ
- **GBDT**: LightGBM、XGBoost、CatBoost
- **CV**: timm、torchvision、albumentations
- **その他**: pandas、numpy、polars

### 実験追跡
- **Weights & Biases (wandb)**: 実験ログ・可視化（オプション）

## 環境構築

### 1. uvのインストール
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 仮想環境の作成
```bash
# 仮想環境構築
uv sync

# 注意: uvは自動的に仮想環境を管理するため、手動でのアクティベートは不要
# コマンドは 'uv run python ...' として実行
```

### 3. Kaggleデータの準備
```bash
cd input/
kaggle competitions download -c {competition_name}
unzip {competition_name}.zip -d ./{competition_name}
rm -f {competition_name}.zip
```

## プロジェクト構成

```
.
├── README.md
├── CLAUDE.md              # このファイル
├── pyproject.toml         # Python依存関係
├── uv.lock               # ロックファイル
├── input/                # コンペティションデータ
│   └── {competition_name}/
├── src/                  # メインコード
│   ├── conf/            # Hydra設定ファイル
│   │   ├── dir/        # データパス設定
│   │   ├── model/      # モデル設定
│   │   └── *.yaml      # メイン設定ファイル
│   ├── datasets/        # データセット定義
│   ├── models/         # モデル定義
│   ├── utils/          # ユーティリティ関数
│   ├── 01-train.py     # トレーニングスクリプト
│   └── 02-infer.py     # 推論スクリプト
├── output/              # 実験結果
│   ├── train/          # トレーニング結果
│   ├── infer/          # 推論結果
│   └── upload/         # Kaggle提出用
├── multirun/            # Hydraマルチラン結果
└── notebook/            # Jupyter notebooks
```

## よく使用するコマンド

### 基本的な実験実行
```bash
# 基本トレーニング（単一fold）
uv run python src/01-train.py fold=0

# 全fold訓練
uv run python src/01-train.py --multirun fold=0,1,2,3,4

# 異なるモデルで実験
uv run python src/01-train.py model=EffcientNet

# 推論実行
uv run python src/02-infer.py

# ハイパーパラメータ変更
uv run python src/01-train.py lr=1e-3 train_batch_size=32
```

### パッケージ管理
```bash
# パッケージ追加
uv add "numpy==1.26.4"
uv add "new-package>=1.0.0"

# パッケージ削除
uv remove package-name

# ロックファイル更新
uv lock
```

### Kaggle提出関連
```bash
# モデルデータセット作成
kaggle datasets init -p output/upload
# dataset-metadata.jsonを編集後
kaggle datasets create -p output/upload --dir-mode zip

# データセット更新
kaggle datasets version -p output/upload -m 'update message' --dir-mode zip

# リポジトリデータセット作成
kaggle datasets init -p src
kaggle datasets create -p src --dir-mode zip

# Kaggle Kernelの取得（推論用）
kaggle kernels pull {username}/{kernel-name}
```

## ワークフロー

### 0. 環境構築
```bash
# uvインストール（必要な場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境構築
uv sync

# Kaggleデータ取得
cd input/
kaggle competitions download -c {competition_name}
unzip {competition_name}.zip -d ./{competition_name}
rm -f {competition_name}.zip
cd ..
```

### 1. データ準備・EDA
```bash
# EDA実行（notebook/で）
uv run jupyter lab notebook/
```

### 2. ベースライン構築
```bash
# 単一モデル実験
uv run python src/01-train.py fold=0

# 全fold訓練
uv run python src/01-train.py --multirun fold=0,1,2,3,4
```

### 3. モデル改善
```bash
# 異なるモデルアーキテクチャ
uv run python src/01-train.py model=ResNet50

# ハイパーパラメータ調整
uv run python src/01-train.py lr=1e-3 train_batch_size=64

# データ拡張実験
uv run python src/01-train.py augmentation=strong
```

### 4. 推論・提出（ローカル環境）
```bash
# 推論実行
uv run python src/02-infer.py

# Kaggleデータセット作成・提出
# README.mdの手順に従う
```

### 5. Kaggle環境での推論・提出
```python
# Kaggle Notebook環境での実行例
# 1. 必要ライブラリのインストール
!pip install hydra-core --no-index --find-links=/kaggle/input/ex-library

# 2. リポジトリコードへのパス追加
import sys
sys.path.append("/kaggle/input/{competition-name}-repo")

# 3. 推論実行
!python /kaggle/input/{competition-name}-repo/02-infer.py \
    dir=kaggle \
    model_dir=/kaggle/input/{competition-name}-models \
    fold=0

# 4. 提出ファイルの移動
import glob
import shutil

pattern = '/kaggle/working/output/infer/**/submission.csv'
files = glob.glob(pattern, recursive=True)
if files:
    shutil.move(files[0], '/kaggle/working/submission.csv')
    print(f"Moved {files[0]} to submission.csv")

# 5. クリーンアップ
!rm -rf output
```

## 設定管理

### Hydra設定構造
- `src/conf/{task}.yaml`: メイン設定ファイル
- `src/conf/dir/`: データパス設定（local.yaml, kaggle.yaml）
- `src/conf/model/`: モデル別設定

### 設定オーバーライド例
```bash
# 学習率変更
uv run python src/01-train.py lr=1e-3

# バッチサイズ変更
uv run python src/01-train.py train_batch_size=64

# 複数パラメータ変更
uv run python src/01-train.py lr=1e-3 num_epochs=30 fold=0
```

## Kaggleデータセット管理

このプロジェクトでは2つのKaggleデータセットを管理します：

### 1. モデルデータセット（{competition-name}-models）
```bash
# 学習済みモデルを準備
cp -r output/train/{timestamp}/*.pth output/upload/models/
cp output/train/{timestamp}/train.log output/upload/

# データセット作成
cd output/upload
kaggle datasets init -p .
# dataset-metadata.jsonを編集
kaggle datasets create -p . --dir-mode zip

# 更新
kaggle datasets version -p . -m 'add new models' --dir-mode zip
```

### 2. リポジトリデータセット（{competition-name}-repo）
```bash
# srcディレクトリをデータセット化
cd src
kaggle datasets init -p .
# dataset-metadata.jsonを編集
kaggle datasets create -p . --dir-mode zip

# 更新（コード変更時）
kaggle datasets version -p . -m 'update inference code' --dir-mode zip
```

### dataset-metadata.json例
```json
{
  "title": "{competition-name}-models",
  "id": "{username}/{competition-name}-models",
  "licenses": [{"name": "CC0-1.0"}]
}
```

## デバッグ・トラブルシューティング

### よくある問題

1. **CUDA out of memory**
   - `train_batch_size`を小さくする
   - `valid_batch_size`を小さくする

2. **設定ファイルが見つからない**
   - `src/conf/`ディレクトリ構造を確認
   - YAML形式の構文エラーをチェック

3. **データパスエラー**
   - `src/conf/dir/local.yaml`のパスを環境に合わせて修正

4. **Kaggle環境での実行エラー**
   - データセットが見つからない：データセットがアタッチされているか確認
   - ライブラリインポートエラー：外部ライブラリデータセットを確認
   - パス関連エラー：`/kaggle/input/`配下のパス構造を確認

### ログの確認
```bash
# 最新の訓練ログ確認
find output/train -name "*.log" -exec ls -la {} + | tail -5

# エラー検索
grep -r "Error" output/train/*/train.log | tail -10
```

## パフォーマンスチューニング

### 一般的な高速化
- `num_workers`調整（データローダー並列化）
- `pin_memory=True`（GPU転送高速化）
- AMP（Automatic Mixed Precision）使用

### メモリ効率化
- バッチサイズの調整
- `torch.cuda.empty_cache()`の適切な使用
- データ前処理の最適化

## 実験管理のベストプラクティス

1. **systematic実験**
   - 1つずつパラメータを変更
   - ベースラインとの比較を明確に

2. **再現性確保**
   - seedの固定
   - 設定ファイルの保存
   - 結果の記録

3. **効率的なイテレーション**
   - 小さなデータセットでの先行実験
   - 段階的な複雑化
   - early stoppingの活用

## コンペティションタイプ別の対応

### 表形式データ
- GBDT（LightGBM, XGBoost, CatBoost）を活用
- 特徴量エンジニアリングに重点
- クロスバリデーション戦略の検討

### 画像データ
- CNN（ResNet, EfficientNet, Vision Transformer）
- データ拡張（albumentations）
- 事前学習モデルの活用（timm）

### 音声データ
- メルスペクトログラム変換
- 音声データ拡張
- CNN/RNNの組み合わせ

### テキストデータ
- 事前学習言語モデル（BERT, RoBERTa）
- トークン化とベクトル化
- ファインチューニング戦略

## 備考

- このテンプレートは様々なタイプのKaggleコンペティションに対応できるよう設計されている
- 新しいコンペティションに適用する際は、`src/conf/dir/local.yaml`のパス設定と`src/datasets/`, `src/models/`内の定義を調整する
- GPU環境での実行を前提としているため、CPU環境では設定の調整が必要

### Kaggle環境での注意点
- Kaggle Notebookでは2つのデータセット（repo用とmodels用）の管理が重要
- 推論実行時は`dir=kaggle`パラメータでKaggle環境用の設定を使用
- 提出ファイルは自動的に`/kaggle/working/submission.csv`に配置される必要がある
- メモリ不足を避けるため、不要なファイルは適切にクリーンアップする
- **重要**: Kaggle環境では`uv`が利用できないため、通常の`python`コマンドを使用する
