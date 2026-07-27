# SGW2026

このリポジトリは、重み付き符号付きネットワーク (WSN) に対する
Fairness-Goodness Algorithm (FGA) の実装と実行例を含みます。

実装本体は `fga/fga.py` です。

## できること

- ノードごとの Fairness スコア (評価者としての一貫性)
- ノードごとの Goodness スコア (被評価者としての信頼性)
- Bitcoin OTC データセット (SNAP) を入力として実行
- 小規模な toy example を使った動作確認
- 評価回数・評価期間を使う weighted 版 FGA の実行
- 分布ヒストグラムと toy グラフ可視化の PNG 出力

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
`--algorithm weighted` を指定した場合は、時系列属性付きの `toy_example_temporal` が使われます。

```bash
python fga/fga.py
```

`weighted` の toy example では、`--visualize-toy` を併用してネットワーク図を保存できます。`weighted` の場合は時系列ラベルも同時に表示され、保存ファイル名は通常の toy 可視化と区別されます。

```bash
python fga/fga.py --algorithm weighted --visualize-toy
```

### 3) 内蔵 toy データセット名を `--dataset` で指定して実行

`--dataset` にはファイルパスだけでなく、次の内蔵 toy データセット名も指定できます。

- `toy_example`
- `toy_example_temporal`
- `toy_example_temporal_p_long`
- `toy_example_temporal_p_short`

例:

```bash
python fga/fga.py --dataset toy_example_temporal_p_long --algorithm weighted --visualize-toy
```

## 主な引数

- `--dataset` : 入力 CSV または CSV.GZ のパス
- `--algorithm` : `basic` または `weighted` (既定値: `weighted`)
- `--eps` : 収束判定しきい値 (既定値: 0.001)
- `--top` : 上位/下位表示件数 (既定値: 10)
- `--max-iter` : 最大反復回数 (既定値: 50)
- `--save-distributions` : Goodness/Fairness の分布ヒストグラムを PNG 保存
- `--visualize-toy` : toy グラフを PNG 保存 (ノード色=Goodness, ノードサイズ=Fairness)。時系列属性がある toy の場合はエッジラベルに重みと時刻を同時表示します。
- `--output-dir` : 分布画像と toy 可視化画像の保存先ディレクトリ (既定値: `fga/`)

## データセット概要

このリポジトリで利用できるデータセットは、外部データセット 1 種類と内蔵 toy データセット 4 種類です。

- `soc-sign-bitcoinotc.csv`
	- 形式: CSV (カンマ区切り)
	- 由来: SNAP の Bitcoin OTC 署名付き信頼ネットワーク
	- 想定カラム: `source, target, rating, time`
	- 用途: 実データに対する FGA (basic/weighted) の実行
- `toy_example`
	- 形式: 内蔵 toy データ (非時系列)
	- 用途: 最小構成でのアルゴリズム挙動確認
- `toy_example_temporal`
	- 形式: 内蔵 toy データ (時系列付き)
	- 特徴: `time` 属性を [0, 1] に正規化
	- 用途: weighted 版 FGA の基本挙動確認
- `toy_example_temporal_p_long`
	- 形式: 内蔵 toy データ (時系列付き)
	- 特徴: 観測期間を長めに分散させた時系列
	- 用途: 評価期間が長いケースの比較
- `toy_example_temporal_p_short`
	- 形式: 内蔵 toy データ (時系列付き)
	- 特徴: 観測時刻が中盤に集中した時系列
	- 用途: 評価期間が短いケースの比較

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

内蔵 toy データセット名を指定して可視化する例:

```bash
python fga/fga.py --dataset toy_example_temporal_p_short --algorithm weighted --visualize-toy --output-dir plots
```

保存ファイル名は実行条件に応じて変わります。例えば `weighted` と `basic` を比較する場合、次のような名前で保存されます。

- `soc-sign-bitcoinotc_weighted_eps-0.001_max-iter-50_distributions.png`
- `soc-sign-bitcoinotc_basic_eps-0.001_max-iter-50_distributions.png`

toy example 可視化を保存した場合は、アルゴリズムに応じて次のような名前になります。

- `toy-example_basic_eps-0.001_max-iter-50_graph.png`
- `toy-example_weighted_eps-0.001_max-iter-50_temporal_graph.png`

## 出力の見方

実行すると次の 3 種類が表示されます。

1. Goodness 上位ノード
2. Goodness 下位ノード (荒らし/詐欺的ノード候補)
3. Fairness 下位ノード (評価が不安定な評価者候補)

各行にはノード ID と `f` (Fairness)、`g` (Goodness) が表示されます。

## 実装上の補足

- `--algorithm basic` のときは `compute_fairness_goodness` を実行し、データセット読み込みに `load_bitcoin_otc` を使います。
- `--algorithm weighted` のときは `compute_fairness_goodness_with_evaluation_weights` を実行し、データセット読み込みに `load_bitcoin_otc_temporal` を使います。
- `--dataset` を指定しない `--algorithm weighted` のときは、`toy_example_temporal` を使って時系列付き toy example を実行します。
- weighted 版では Fairness に「評価回数の正規化値 × 評価期間の正規化値」を掛け、Goodness にはノード自身の評価期間の正規化値を掛けます。
- 正規化にはシグモイド関数 $\sigma(x)=1/(1+e^{-x})$ を使います。
- `load_bitcoin_otc` は重みのみ (`weight`) を設定し、`load_bitcoin_otc_temporal` は重み (`weight`) に加えて時刻 (`time`) も [0, 1] に正規化して設定します。
- エッジ属性が欠損している場合、`weight` は 0.0 として扱われます。

## テスト

`tests/` 配下にテストコードがあります。環境に応じて次のように実行してください。

```bash
python -m pytest
```

## 追記

`soc-sign-bitcoinotc.csv` は、Bitcoin OTC ユーザ間の評価関係を表す符号付き有向ネットワークデータで、各行は「評価者ノード・被評価者ノード・評価値・タイムスタンプ」に対応します。本実装では `rating` をエッジ重みとして利用し、weighted 版では `time` を [0, 1] に正規化して時系列情報として扱います。

また、`fga.py` には一部、生成 AI を用いて作成・補助されたコードが含まれています。
