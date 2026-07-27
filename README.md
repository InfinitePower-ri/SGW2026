# SGW2026

このリポジトリは、重み付き符号付きネットワーク (WSN) に対する
Fairness-Goodness Algorithm (FGA) の実装と実行例を含みます。

実装本体は `fga/fga.py` です。

## できること

- ノードごとの Fairness スコア (評価者としての一貫性)
- ノードごとの Goodness スコア (被評価者としての信頼性)
- Bitcoin OTC データセット (SNAP) を入力として実行
- 小規模な toy example を使った動作確認

## 前提環境

- Python 3.9 以上を推奨
- 必要パッケージ:
	- networkx
	- matplotlib

インストール例:

```bash
pip install networkx matplotlib
```

## 実行方法

### 1) Bitcoin OTC データセットで実行

`fga/fga.py` は、相対パスの場合に `fga` ディレクトリ基準でデータセットを読み込みます。
そのため、同梱データセットを使う場合は次のコマンドで実行できます。

```bash
python fga/fga.py --dataset soc-sign-bitcoinotc.csv
```

`.gz` ファイルも読み込み可能です。

### 2) toy example で実行

`--dataset` を指定しない場合、内蔵の toy example が実行されます。

```bash
python fga/fga.py
```

## 主な引数

- `--dataset` : 入力 CSV または CSV.GZ のパス
- `--algorithm` : `basic` または `weighted` (既定値: `weighted`)
- `--eps` : 収束判定しきい値 (既定値: 0.001)
- `--top` : 上位/下位表示件数 (既定値: 10)
- `--max-iter` : 最大反復回数 (既定値: 50)
- `--save-distributions` : Goodness/Fairness の分布ヒストグラムを PNG 保存
- `--visualize-toy` : toy example 実行時にネットワーク図を PNG 保存 (ノード色=Goodness, ノードサイズ=Fairness, エッジラベル=重み)
- `--output-dir` : 分布画像の保存先ディレクトリ (既定値: `fga/`)

例:

```bash
python fga/fga.py --dataset soc-sign-bitcoinotc.csv --eps 0.001 --top 20 --max-iter 100
```

分布画像も保存する例:

```bash
python fga/fga.py --dataset soc-sign-bitcoinotc.csv --algorithm weighted --save-distributions
```

toy example のネットワーク図を保存する例:

```bash
python fga/fga.py --algorithm weighted --visualize-toy
```

保存ファイル名は実行条件に応じて変わります。例えば `weighted` と `basic` を比較する場合、次のような名前で保存されます。

- `soc-sign-bitcoinotc_weighted_eps-0.001_max-iter-50_distributions.png`
- `soc-sign-bitcoinotc_basic_eps-0.001_max-iter-50_distributions.png`

toy example 可視化を保存した場合は、次のような名前になります。

- `toy-example_weighted_eps-0.001_max-iter-50_graph.png`

## 出力の見方

実行すると次の 3 種類が表示されます。

1. Goodness 上位ノード
2. Goodness 下位ノード (荒らし/詐欺的ノード候補)
3. Fairness 下位ノード (評価が不安定な評価者候補)

各行にはノード ID と `f` (Fairness)、`g` (Goodness) が表示されます。

## 実装上の補足

- `--algorithm basic` のときは `compute_fairness_goodness` を実行し、データセット読み込みに `load_bitcoin_otc` を使います。
- `--algorithm weighted` のときは `compute_fairness_goodness_with_evaluation_weights` を実行し、データセット読み込みに `load_bitcoin_otc_temporal` を使います。

## テスト

`tests/` 配下にテストコードがあります。環境に応じて次のように実行してください。

```bash
python -m pytest
```
