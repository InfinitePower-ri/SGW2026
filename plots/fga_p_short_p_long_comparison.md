# FGA比較結果（p_short / p_long, basic / weighted）

## 実行条件
- Python: .venv (3.10.5)
- 実行方式: fga/fga.py の実装関数を直接呼び出し
- 対象データセット: toy_example_temporal_p_short, toy_example_temporal_p_long
- アルゴリズム: basic, weighted
- eps: 0.001
- max_iter: 50
- top: 20（比較計算自体は全ノードで実施）

## 実行日時
- 2026-07-27 22:33:20

## 出力ファイル
- fga_p_short_p_long_comparison.csv
  - basic_vs_weighted: 各データセット内で weighted-basic を比較
  - p_short_vs_p_long_weighted: weighted 条件で p_short-p_long を比較
