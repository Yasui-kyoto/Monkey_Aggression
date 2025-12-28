import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 個体間関係の推察
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, nx, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## グルーミング
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1. データの読み込み
    """)
    return


@app.cell
def _(pd):
    # CSVの1列目をインデックス（起点個体）として読み込みます
    df = pd.read_csv("data/Kojima_gr_combination.csv", index_col=0)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. 有向グラフの構築 (有向グラフ: Directed Graph)
    """)
    return


@app.cell
def _(df, nx, pd):
    G = nx.DiGraph()

    # データフレームをループしてエッジ（つながり）を追加
    for actor in df.index:
        for receiver in df.columns:
            weight = df.loc[actor, receiver]
            if pd.notna(weight) and weight > 0:
                G.add_edge(actor, receiver, weight=float(weight))
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. 各種指標の計算（Rのigraphに近い算出）
    """)
    return


@app.cell
def _(G):
    # ノードサイズ：重み付きの次数（Weighted Strength = グルーミング頻度の合計）
    node_strength = dict(G.degree(weight='weight'))
    # 中心性：固有ベクトル中心性など（オプション）
    # centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    return (node_strength,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. レイアウトと描画設定
    """)
    return


@app.cell
def _(G, mo, node_strength, nx, plt):
    plt.figure(figsize=(12, 10))
    pos = nx.circular_layout(G)  # 円状レイアウト

    # エッジの太さを重みに応じて調整
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [w * 1.5 for w in weights]  # 見映えのために係数を掛ける

    # ノードのサイズをStrengthに基づいて調整
    node_sizes = [node_strength[node] * 300 for node in G.nodes()]

    # 描画の実行
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='orange', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', 
                           alpha=0.5, arrowsize=15, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.title("Grooming Network (Kojima Group)", fontsize=15)
    plt.axis('off')

    # Marimoで表示するための出力
    chart = plt.gca()
    mo.as_html(chart)
    return


if __name__ == "__main__":
    app.run()
