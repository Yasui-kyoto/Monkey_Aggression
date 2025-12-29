import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf



class GroomAim1:
    def __init__(self):
        pass
    
    
    
    def plot_behavior_scatter(self, df, y_column, x_column='delta_time', hue_column='behavior'):
        """
        指定されたデータフレームから散布図を作成する関数。

        Parameters:
        df (pd.DataFrame): データセット
        y_column (str): 縦軸にするカラム名 ('delta_face' や 'delta_nose')
        x_column (str): 横軸にするカラム名 (デフォルトは 'delta_time')
        hue_column (str): 色分けの基準とするカラム名 (デフォルトは 'behavior')
        """
        plt.figure(figsize=(10, 6))
        
        # behaviorごとの色を固定する辞書（brightパレットの順序に準拠）
        color_palette = {'BL': '#023eff', 'grooming': '#ff7c00', 'groomed': '#1ac938'}

        # seabornのscatterplotを使用すると、behaviorごとの色分けが自動で行われます
        sns.scatterplot(
            data=df, 
            x=x_column, 
            y=y_column, 
            hue=hue_column, 
            palette=color_palette,
            alpha=0.7           # 点の透明度
        )
        
        position_dict = {'delta_face': 'Face Temperature', 'delta_nose': 'Nose Temperature'}

        plt.title(f'Time Series Analysis:Scatter plot of {position_dict[y_column]} Change')
        plt.xlabel('Time from the starting point (s)')
        plt.ylabel('Temperature change (°C)')
        plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc='upper left') # 凡例を外側に配置
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()



    def plot_highlight_behavior(self, df, target_behavior, y_column, x_column='delta_time'):
        """
        特定のbehaviorだけを色付けし、それ以外を灰色で表示する関数。
        """
        plt.figure(figsize=(10, 6))
        
        # behaviorごとの色を固定する辞書（brightパレットの順序に準拠）
        color_palette = {'BL': '#023eff', 'grooming': '#ff7c00', 'groomed': '#1ac938'}
        
        # もしリストにないbehaviorが指定された場合のデフォルト色
        target_color = color_palette.get(target_behavior, 'red')

        # 1. ターゲット以外のデータを灰色でプロット
        other_df = df[df['behavior'] != target_behavior]
        sns.scatterplot(
            data=other_df,
            x=x_column,
            y=y_column,
            color='gray',
            alpha=0.25,
            label='Others'
        )

        # 2. ターゲットのデータだけを色付きで重ねてプロット
        target_df = df[df['behavior'] == target_behavior]
        sns.scatterplot(
            data=target_df,
            x=x_column,
            y=y_column,
            color=target_color, # 固定色を使用
            alpha=0.8,
            label=target_behavior
        )

        position_dict = {'delta_face': 'Face Temperature', 'delta_nose': 'Nose Temperature'}
        plt.title(f'Highlight: {target_behavior} ({position_dict[y_column]})')
        plt.xlabel('Time from the starting point (s)')
        plt.ylabel('Temperature change (°C)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()