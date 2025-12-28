import marimo

__generated_with = "0.10.14" # バージョンは任意
app = marimo.App()


@app.cell
def __():
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo
    return mo, nx, pd, plt, np


@app.cell
def __(pd):
    # パス設定
    path_matrix = r"C:\Users\yyu33\Downloads\Monkey_Aggression\data\Kojima_gr_combination.csv"
    path_attr = r"C:\Users\yyu33\Downloads\Monkey_Aggression\data\monkey_data.csv"
    
    df_matrix = pd.read_csv(path_matrix, index_col=0)
    df_attr = pd.read_csv(path_attr)
    return df_attr, df_matrix


@app.cell
def __(df_attr, df_matrix, mo, nx, plt):
    # グラフ構築
    G = nx.DiGraph()
    for actor in df_matrix.index:
        for receiver in df_matrix.columns:
            weight = df_matrix.loc[actor, receiver]
            if not plt.np.isnan(weight) and weight > 0:
                G.add_edge(actor, receiver, weight=float(weight))

    # 性別色分け
    node_colors = []
    for node in G.nodes():
        row = df_attr[df_attr['name'] == node]
        if not row.empty:
            sex = row.iloc[0]['sex']
            node_colors.append('orange' if sex == 'f' else 'steelblue' if sex == 'm' else 'gray')
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
    
    ax.set_title("Grooming Network")
    ax.axis('off')
    
    # marimoで表示
    display_output = mo.as_html(fig)
    return G, display_output


@app.cell
def __(display_output):
    display_output
    return


if __name__ == "__main__":
    app.run()