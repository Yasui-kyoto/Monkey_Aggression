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



    def interpolate_samples(self, df, time_limit=300):
        """
        全サンプルを1秒単位で線形補間し、解析用のベースデータを作成する。
        """
        interpolated_list = []

        print(f"Interpolating {df['sampling_id'].nunique()} samples...")

        for sid, group in df.groupby('sampling_id'):
            new_index = np.arange(0, time_limit + 1)
            temp_group = group.drop_duplicates(subset='delta_time').set_index('delta_time')

            # 器の作成と数値補間
            resampled = temp_group.reindex(new_index)
            resampled = resampled.infer_objects(copy=False)
            resampled = resampled.interpolate(method='linear')
            
            # 非数値列の補完（前後埋め）
            resampled = resampled.ffill().bfill()
            
            resampled['sampling_id'] = sid
            resampled.index.name = 'delta_time'
            interpolated_list.append(resampled.reset_index())

        full_df = pd.concat(interpolated_list).reset_index(drop=True)
        print("Interpolation completed.")
        return full_df
    
    
    
    def plot_smoothed_behavior_comparison(self, interpolated_df, y_column='delta_nose', target_behaviors=None):
        """
        内挿済みデータを使用して帯プロットを作成する。
        target_behaviors: ['BL', 'grooming'] のように指定するとそのペアのみプロット。
        """
        plot_df = interpolated_df.copy()

        # 特定のペアのみに絞り込むオプション
        if target_behaviors:
            plot_df = plot_df[plot_df['behavior'].isin(target_behaviors)]

        if plot_df.empty:
            print("指定された行動のデータが存在しません。")
            return

        plt.figure(figsize=(10, 6))
        
        # 色指定（これまでの設定を維持）
        color_palette = {'BL': '#023eff', 'grooming': '#ff7c00', 'groomed': '#1ac938'}

        # 信頼区間（帯）の描画
        sns.lineplot(
            data=plot_df,
            x='delta_time',
            y=y_column,
            hue='behavior',
            palette=color_palette,
            errorbar=('ci', 95),
            n_boot=500
        )

        title_suffix = f"({', '.join(target_behaviors)})" if target_behaviors else "(All Behaviors)"
        plt.title(f'Temperature Trend {title_suffix}\nInterpolated 1s intervals | {y_column}')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature Change (°C)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='-')

        plt.tight_layout()
        plt.show()
        
        
        
        
class GroomAim1LinearTests:
    def test_behavior_pair_comparison(self, df, behavior_a, behavior_b, y_column='delta_nose'):
        """
        指定した2つのbehavior間で、温度変化の傾きに有意差があるかを線形混合モデルで検定する。
        """
        # 1. 比較対象の2群のみを抽出
        test_df = df[df['behavior'].isin([behavior_a, behavior_b])].copy()

        # 2. 混合モデルの実行（behaviorをカテゴリカル変数として扱う）
        # 固定効果: delta_time, behavior, およびその相互作用
        # 変量効果: sampling_id ごとの切片
        formula = f"{y_column} ~ delta_time * behavior"

        try:
            model = smf.mixedlm(formula, test_df, groups=test_df["sampling_id"])
            result = model.fit()

            print(f"\n========== Pairwise Comparison: {behavior_a} vs {behavior_b} ({y_column}) ==========")

            # 相互作用項（delta_time:behavior[...]）のインデックス名を探す
            interaction_term = [idx for idx in result.pvalues.index if 'delta_time:behavior' in idx]

            if interaction_term:
                term = interaction_term[0]
                p_val = result.pvalues[term]
                coef = result.params[term]

                print(f"Comparison: {behavior_a} と {behavior_b} の傾きの差")
                print(f"Interaction Coef (Diff in Slope): {coef:.6f}")
                print(f"P-value: {p_val:.4f}")

                print("---------------------------------------------------------")
                if p_val < 0.05:
                    print(f"結果: ★ 有意な差があります。")
                elif p_val < 0.1:
                    print(f"結果: △ 有意な傾向があります。")
                else:
                    print(f"結果: 有意な差は認められませんでした。")
                print("---------------------------------------------------------")
            else:
                print("交互作用項の算出に失敗しました。")

            return result

        except Exception as e:
            print(f"モデルの実行中にエラーが発生しました: {e}")
            return None
        
        

    def plot_behavior_trend_comparison(self, df, behavior_a, behavior_b, y_column='delta_nose'):
        """
        2つの行動の温度変化の『勢い（傾き）』の差を可視化する。
        生データではなく、平均的なトレンドと回帰直線を表示。
        """
        # 1. 比較対象の2群を抽出
        plot_df = df[df['behavior'].isin([behavior_a, behavior_b])].copy()

        plt.figure(figsize=(10, 6))

        # 2. カラーパレットの設定（これまでの設定に準拠）
        color_palette = {'BL': '#023eff', 'grooming': '#ff7c00', 'groomed': '#1ac938'}

        # 3. 平均トレンドの描画 (Confidence Interval 95% を影として表示)
        # これにより、個別の点のバラつきを「平均的な帯」として整理できます
        sns.lineplot(
            data=plot_df,
            x='delta_time',
            y=y_column,
            hue='behavior',
            palette=color_palette,
            errorbar=('ci', 95), # 95%信頼区間を影で表示
            alpha=0.4,
            linewidth=1,
            legend=True
        )

        # 4. 回帰直線（傾きそのもの）を重ね書き
        # 検定が「傾き」を見ているので、これこそが「統計学的に見えている線」です
        for beh in [behavior_a, behavior_b]:
            subset = plot_df[plot_df['behavior'] == beh]
            sns.regplot(
                data=subset,
                x='delta_time',
                y=y_column,
                scatter=False,    # 点は描かない
                color=color_palette.get(beh),
                label=f'{beh} Regression Line',
                line_kws={'linestyle': '--', 'linewidth': 3} # 太めの点線で強調
            )

        position_dict = {'delta_face': 'Face', 'delta_nose': 'Nose'}
        plt.title(f'Trend Comparison: {behavior_a} vs {behavior_b}\n(Linear Regression & 95% CI)')
        plt.xlabel('Time from the starting point (s)')
        plt.ylabel(f'{position_dict.get(y_column, y_column)} Temperature change (°C)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        
        
    def plot_final_delta_comparison(self, df, y_column='delta_nose'):
        """
        各サンプルの『最終的な温度変化量』だけを取り出して比較する。
        """
        # 各サンプリングIDの最後のデータポイントだけを抽出
        final_points = df.sort_values('delta_time').groupby('sampling_id').last().reset_index()

        # 色の設定（これまでの散布図と統一）
        color_palette = {'BL': '#023eff', 'grooming': '#ff7c00', 'groomed': '#1ac938'}

        plt.figure(figsize=(8, 6))

        # 1. 箱ひげ図 (アウトライヤーは表示しない設定: showfliers=False)
        # なぜなら、下のstripplotですべての生データを打点するため。
        sns.boxplot(
            data=final_points, 
            x='behavior', 
            y=y_column, 
            order=['BL', 'groomed', 'grooming'], # 順番を固定
            palette=color_palette,
            showfliers=False,
            boxprops=dict(alpha=0.3) # 箱を少し薄くして点を見やすくする
        )

        # 2. 生データ（個々のサンプリングIDの最終値）の打点
        # これが「●」になります
        sns.stripplot(
            data=final_points, 
            x='behavior', 
            y=y_column, 
            order=['BL', 'groomed', 'grooming'],
            palette=color_palette,
            size=6,
            alpha=0.7,
            jitter=True # 点が重ならないように少し左右に散らす
        )

        position_dict = {'delta_face': 'Face', 'delta_nose': 'Nose'}
        plt.title(f'Final {position_dict.get(y_column, y_column)} Temperature Change\n(Last point of each sample)')
        plt.ylabel(f'{position_dict.get(y_column, y_column)} Delta (°C)')
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()