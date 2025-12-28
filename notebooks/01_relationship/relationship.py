import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo
    return mo, nx, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## グルーミングdataと属性dataの読み込み
    """)
    return


@app.cell
def _(pd):
    # パス設定
    path_matrix = r"C:\Users\yyu33\Downloads\Monkey_Aggression\data\Kojima_gr_combination.csv"
    path_attr = r"C:\Users\yyu33\Downloads\Monkey_Aggression\data\monkey_data.csv"

    df_matrix = pd.read_csv(path_matrix, index_col=0)
    df_attr = pd.read_csv(path_attr)
    return df_attr, df_matrix


@app.cell
def _(df_matrix):
    print(df_matrix.head())
    return


@app.cell
def _(df_attr):
    print(df_attr.head())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### グルーミングに基づく関係グラフの構築
    """)
    return


@app.cell
def _(df_attr, df_matrix, mo, nx, plt):
    # グラフ構築
    G = nx.DiGraph()
    for _actor in df_matrix.index:
        for _receiver in df_matrix.columns:
            _weight = df_matrix.loc[_actor, _receiver]
            if not plt.np.isnan(_weight) and _weight > 0:
                G.add_edge(_actor, _receiver, weight=float(_weight))

    # 性別色分け
    node_colors = []
    for _node in G.nodes():
        _row = df_attr[df_attr['name'] == _node]
        if not _row.empty:
            _sex = _row.iloc[0]['sex']
            node_colors.append('orange' if _sex == 'f' else 'steelblue' if _sex == 'm' else 'gray')
        else:
            node_colors.append('gray')

    # 描画
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.circular_layout(G)

    node_strength = dict(G.degree(weight='weight'))
    node_sizes = [node_strength[n] * 300 for n in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, edgecolors='white', ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.3, arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9)

    ax.set_title("Grooming Network (Size = Outgoing + Incoming Grooming)")
    ax.axis('off')

    # marimoで表示
    display_output = mo.as_html(fig)
    return (display_output,)


@app.cell
def _(display_output):
    display_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 攻撃関係
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 攻撃dataの読み込み
    """)
    return


@app.cell
def _(pd):
    path_aggresion_matrix = r"C:\Users\yyu33\Downloads\Monkey_Aggression\data\Kojima_aggression.csv"

    df_aggresion = pd.read_csv(path_aggresion_matrix, index_col=0)
    return (df_aggresion,)


@app.cell
def _(df_aggresion):
    print(df_aggresion.head())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 攻撃関係に基づくグラフの構築
    （ノードサイズは攻撃を仕掛けた回数（Out-degree）を反映）
    """)
    return


@app.cell
def _(df_aggresion, df_attr, mo, nx, plt):
    # 1. 攻撃グラフの構築
    G_agg = nx.DiGraph()
    for _actor in df_aggresion.index:
        for _receiver in df_aggresion.columns:
            _weight = df_aggresion.loc[_actor, _receiver]
            # 欠損値でなく、1回以上の攻撃がある場合
            if not plt.np.isnan(_weight) and _weight > 0:
                G_agg.add_edge(_actor, _receiver, weight=float(_weight))

    # 2. 性別に基づく色分け（既存のロジックを流用）
    node_colors_agg = []
    for _node in G_agg.nodes():
        _row = df_attr[df_attr['name'] == _node]
        if not _row.empty:
            _sex = _row.iloc[0]['sex']
            node_colors_agg.append('orange' if _sex == 'f' else 'steelblue' if _sex == 'm' else 'gray')
        else:
            node_colors_agg.append('gray')

    # 3. 描画の設定
    fig_agg, ax_agg = plt.subplots(figsize=(10, 8))
    pos_agg = nx.circular_layout(G_agg)

    # ノードサイズ：攻撃を仕掛けた数 (Out-degree strength)
    # out_degree(weight='weight') を使用
    node_out_strength = dict(G_agg.out_degree(weight='weight'))
    # サイズが0だと見えないため、最小値（例: 100）を足すかスケーリングを調整します
    node_sizes_agg = [(node_out_strength[n] * 500) + 100 for n in G_agg.nodes()]

    # エッジの太さ（攻撃の激しさ）
    agg_weights = [G_agg[u][v]['weight'] * 1.5 for u, v in G_agg.edges()]

    # 4. 描画実行
    nx.draw_networkx_nodes(G_agg, pos_agg, node_color=node_colors_agg, 
                           node_size=node_sizes_agg, alpha=0.8, edgecolors='white', ax=ax_agg)

    # エッジの色をgroomingと同様に灰色（gray）に変更
    nx.draw_networkx_edges(G_agg, pos_agg, width=agg_weights, edge_color='gray', 
                           alpha=0.4, arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax_agg)

    nx.draw_networkx_labels(G_agg, pos_agg, font_size=9)

    ax_agg.set_title("Aggression Network (Size = Out-going Aggression)", fontsize=14)
    ax_agg.axis('off')

    display_output_agg = mo.as_html(fig_agg)
    return (display_output_agg,)


@app.cell
def _(display_output_agg):
    # 攻撃グラフの表示
    display_output_agg
    return


if __name__ == "__main__":
    app.run()
