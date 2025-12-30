import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats



class GroomAim2:
    def __init__(self):
        self.position_dict = {
            'delta_face': 'Face',
            'corrected_delta_face': 'Face (Corrected)',
            'delta_nose': 'Nose',
            'corrected_delta_nose': 'Nose (Corrected)',
            'nose-face': 'Nose - Face',
            'corrected_nose-face': 'Corrected Nose - Face'
        }
        
        self.effect_name_dict = {
            'rank_direction': 'Rank Direction',
            'kin': 'Kinship',
            'centrality_direction': 'Centrality Direction'
        }
        
        pass
    


    def fix_grooming_records(self, df, mode='grooming'):
        """
        記録ミス（from/toの逆転）を修正し、値をスワップした後に
        空いた列に name 列の値を代入する。
        
        Parameters:
        -----------
        df : pd.DataFrame
        mode : str, 'grooming' or 'groomed'
            'grooming': 'to' に相手名を入れる。
            'groomed' : 'from' に相手名を入れる。
        """
        # 1. 判定基準と、最終的に名前を入れるべきターゲット列を決定
        if mode == 'grooming':
            check_col = 'from'  # fromに値があるのがミス
            target_col = 'to'   # 本来相手の名前があるべき方
        else:
            check_col = 'to'    # toに値があるのがミス
            target_col = 'from' # 本来相手の名前があるべき方
        
        print(f"--- Mode: {mode} (Counterpart column: {target_col}) ---")

        # 2. 記録ミスの sampling_id を特定
        error_ids = df.dropna(subset=[check_col])['sampling_id'].unique().tolist()

        # 3. 値のスワップ処理（ミスがあるIDのみ）
        if error_ids:
            print(f"記録ミスの可能性がある sampling_id: {error_ids}")
            mask = df['sampling_id'].isin(error_ids)
            df.loc[mask, ['from', 'to']] = df.loc[mask, ['to', 'from']].values
            print("入れ替え処理が完了しました。")
        else:
            print("スワップ対象の行は見つかりませんでした。")

        # 4. 空いているターゲット列に name 列の値を貼り付ける（全行対象）
        df.loc[:, check_col] = df['name']
        print(f"{check_col} 列を name 列の値で補完しました。")

        # 結果の確認
        print("\n[処理結果の確認(代表各1行)]")
        print(df[['sampling_id', 'name', 'from', 'to']].drop_duplicates(subset=['sampling_id']).head(10))
            
        return df
    
    

    def add_rank_direction(self, grooming_df, rank_dict):
        """
        グルーミングの方向（順位の高低）を判定して列を追加する。
        オス（Kobu, Nishin, Gure）は常に最上位として扱う。
        """
        # 1. 順位を取得する内部補助関数
        def get_rank(name):
            # 辞書にあればその順位、なければ一旦大きな値（最下位以下）を返す
            return rank_dict.get(name, 999)

        # 2. 各行に対して判定を行う
        def judge_direction(row):
            # 名前が取得できない（NaN）場合は判定不可
            if pd.isna(row['from']) or pd.isna(row['to']):
                return 'unknown'
            
            rank_from = get_rank(row['from'])
            rank_to = get_rank(row['to'])

            # 順位の数値が小さいほど「高順位」
            if rank_from < rank_to:
                return 'high_to_low'  # 高順位から低順位へ
            elif rank_from > rank_to:
                return 'low_to_high'  # 低順位から高順位へ
            else:
                return 'equal'        # 同順位（オス同士、またはランクが同じメス同士）

        # 新しい列 'rank_direction' を作成
        # .apply(axis=1) で1行ずつ判定を回す
        results = grooming_df.apply(judge_direction, axis=1)
        grooming_df.loc[:, 'rank_direction'] = results

        print("順位方向の判定が完了しました。")
        print(grooming_df['rank_direction'].value_counts()) # 内訳を表示
        
        return grooming_df
    
    
    
    def add_centrality_direction(self, grooming_df, centrality_dict):
        """
        中心性（Centrality）の高低に基づいたグルーミングの方向を判定して列を追加する。
        """
        # 1. 中心性を取得する内部補助関数
        def get_centrality(name):
            # 辞書にあればその値、なければ最小値（0.0）を返す
            return centrality_dict.get(name, 0.0)

        # 2. 各行に対して判定を行う
        def judge_direction(row):
            # 名前が欠損している場合は判定不可
            if pd.isna(row['from']) or pd.isna(row['to']):
                return 'unknown'
            
            c_from = get_centrality(row['from'])
            c_to = get_centrality(row['to'])

            # 中心性の数値が大きいほど「上位（High）」と判定
            if c_from > c_to:
                return 'high_to_low'  # 中心性が高い個体から低い個体へ
            elif c_from < c_to:
                return 'low_to_high'  # 中心性が低い個体から高い個体へ
            else:
                return 'equal'        # 中心性が同じ

        # 新しい列 'centrality_direction' を作成
        # SettingWithCopyWarning を避けるため .loc を使用
        results = grooming_df.apply(judge_direction, axis=1)
        grooming_df.loc[:, 'centrality_direction'] = results

        print("中心性方向の判定が完了しました。")
        print(grooming_df['centrality_direction'].value_counts())
        
        return grooming_df
    
    
    
    def plot_kinship_comparison(self, groom_df, behavior_type='groomed', y_column='corrected_nose-face'):
        """
        血縁関係（kin: yes vs no）に基づいた温度変化の比較プロットを作成する。
        
        Parameters:
        -----------
        groom_df : pd.DataFrame
            解析対象のデータフレーム
        behavior_type : str, 'grooming' or 'groomed'
            タイトル表示に使用する行動種別
        y_column : str
            縦軸に使用するカラム名
        """
        # 1. データのコピーとフィルタリング
        # kin列が 'yes' または 'no' のデータを抽出し、欠損値を除外
        valid_kin = ['yes', 'no']
        plot_df = groom_df[groom_df['kin'].isin(valid_kin)].copy()

        if plot_df.empty:
            print(f"警告: {behavior_type} において血縁情報（kin）を持つデータが存在しません。")
            return

        plt.figure(figsize=(10, 6))
        
        # 2. 色指定
        # Kin (yes) は親密さを表す暖色系、Non-kin (no) は寒色系などで設定
        kin_palette = {'yes': '#e41a1c', 'no': '#377eb8'}

        # 3. 帯プロット（信頼区間付き平均線）の描画
        sns.lineplot(
            data=plot_df,
            x='delta_time',
            y=y_column,
            hue='kin',
            hue_order=['yes', 'no'], # 凡例の順序を固定
            palette=kin_palette,
            errorbar=('ci', 95),      # 95%信頼区間
            n_boot=500
        )

        # 4. タイトルとラベルの動的設定
        plt.title(f'Temperature Dynamics: {behavior_type.capitalize()}\n'
                  f'Comparison by Kinship | {y_column}')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature Change (°C)')
        
        # 5. 装飾
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axhline(0, color='black', linewidth=1, linestyle='-')
        
        # 凡例のラベルを分かりやすく調整
        plt.legend(title='Kinship (Related)', loc='upper right', labels=['Kin (Yes)', 'Non-kin (No)'])

        plt.tight_layout()
        plt.show()
    
    
    
    def plot_direction_comparison(
        self, 
        groom_df, 
        target_cols='rank_direction',
        behavior_type='groomed', 
        y_column='corrected_nose-face'
    ):
        """
        方向（high_to_low vs low_to_high）に基づいた温度変化の比較プロットを作成する。
        
        Parameters:
        -----------
        groom_df : pd.DataFrame
            解析対象のデータフレーム
        behavior_type : str, 'grooming' or 'groomed'
            タイトル表示に使用する行動種別
        y_column : str
            縦軸に使用するカラム名
        """
        # 1. データのコピーと方向の絞り込み
        # 'equal' や 'unknown' を除外し、有効な方向のみを抽出
        valid_directions = ['high_to_low', 'low_to_high']
        plot_df = groom_df[groom_df[target_cols].isin(valid_directions)].copy()

        if plot_df.empty:
            print(f"警告: {behavior_type} において判定可能な方向のデータが存在しません。")
            return

        plt.figure(figsize=(10, 6))
        
        # 2. 色指定
        # high_to_low (上位から下位へ) と low_to_high (下位から上位へ)
        direction_palette = {'high_to_low': '#1f77b4', 'low_to_high': '#d62728'}

        # 3. 帯プロット（信頼区間付き平均線）の描画
        sns.lineplot(
            data=plot_df,
            x='delta_time',
            y=y_column,
            hue=target_cols,
            palette=direction_palette,
            errorbar=('ci', 95),  # 95%信頼区間を影として表示
            n_boot=500
        )

        # 4. タイトルとラベルの動的設定
        # behavior_type をタイトルに反映
        plt.title(f'Temperature Dynamics: {behavior_type.capitalize()}\n'
                  f'Comparison by {self.effect_name_dict[target_cols]} | {y_column}')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature Change (°C)')
        
        # 5. 装飾
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axhline(0, color='black', linewidth=1, linestyle='-')
        plt.legend(title=f'{self.effect_name_dict[target_cols]}', loc='upper right')

        plt.tight_layout()
        plt.show()
        
        
        
    def run_cluster_based_replacement_test(
        self, 
        groom_df, 
        target_cols='rank_direction',
        target_values=['high_to_low', 'low_to_high'],
        y_column='corrected_nose-face', 
        n_permutations=1000, 
        p_threshold=0.05
    ):
        """
        target_colsが温度変化に与える影響を検定する。
        """
        # 1. データの準備
        # 分析対象の2つのグループを抽出
        data_a = groom_df[groom_df[target_cols] == target_values[0]]
        data_b = groom_df[groom_df[target_cols] == target_values[1]]
        times = sorted(groom_df['delta_time'].unique())
        
        def create_matrix(sub_df):
            # 縦にサンプル(sampling_id)、横に時間(delta_time)の行列を作成
            matrix = sub_df.pivot(index='sampling_id', columns='delta_time', values=y_column)
            # 全ての時間点にデータが揃っているサンプルのみを使用
            return matrix.dropna().values

        matrix_a = create_matrix(data_a)
        matrix_b = create_matrix(data_b)
        
        if len(matrix_a) < 3 or len(matrix_b) < 3:
            print(f"サンプル数が少なすぎるため検定をスキップします。(a: {len(matrix_a)}, b: {len(matrix_b)})")
            return None

        # 2. 実データの t値計算とクラスター特定
        # 対応なしのt検定
        t_obs, _ = stats.ttest_ind(matrix_a, matrix_b, axis=0)
        df_degree = len(matrix_a) + len(matrix_b) - 2
        thresh_t = stats.t.ppf(1 - p_threshold / 2, df_degree)
        
        def find_clusters(t_values, threshold_t):
            clusters = []
            current_indices = []
            for i, t in enumerate(t_values):
                if abs(t) > threshold_t:
                    current_indices.append(i)
                else:
                    if current_indices:
                        clusters.append(current_indices)
                        current_indices = []
            if current_indices:
                clusters.append(current_indices)
            return clusters

        obs_clusters = find_clusters(t_obs, thresh_t)
        obs_cluster_stats = [np.abs(np.sum(t_obs[c])) for c in obs_clusters]

        # 3. 置換検定（帰無分布の作成）
        print(f"Running {n_permutations} permutations for Rank Direction...")
        combined_matrix = np.vstack([matrix_a, matrix_b])
        n_a = len(matrix_a)
        null_max_clusters = []

        for _ in range(n_permutations):
            indices = np.random.permutation(len(combined_matrix))
            # ラベルをシャッフルしてt値を再計算
            t_rand, _ = stats.ttest_ind(combined_matrix[indices[:n_a]], combined_matrix[indices[n_a:]], axis=0)
            
            rand_clusters = find_clusters(t_rand, thresh_t)
            rand_stats = [np.abs(np.sum(t_rand[c])) for c in rand_clusters]
            null_max_clusters.append(np.max(rand_stats) if rand_stats else 0)

        # 4. 結果の判定
        null_dist = np.array(null_max_clusters)
        significant_periods = []
        
        print(f"\n--- Cluster-based Permutation Result: High-to-Low vs Low-to-High ---")
        if not obs_clusters:
            print("閾値を超えるクラスターは検出されませんでした。")
        else:
            for i, c_indices in enumerate(obs_clusters):
                c_stat = obs_cluster_stats[i]
                # このクラスターの統計量がシャッフル分布の何％以上か
                c_p_value = np.sum(null_dist >= c_stat) / n_permutations
                start_s = times[c_indices[0]]
                end_s = times[c_indices[-1]]

                print(f"Cluster {i+1}: {start_s}s - {end_s}s | Sum(t): {c_stat:.2f} | p = {c_p_value:.4f}")
                if c_p_value < p_threshold:
                    print(f"  => ★ 有意差あり")
                    significant_periods.append((start_s, end_s))

        # 5. 可視化
        plt.figure(figsize=(10, 5))
        plt.plot(times, t_obs, label=f'Observed t-value ({target_values[0]} vs {target_values[1]})', color='black')
        plt.axhline(thresh_t, color='red', linestyle='--', alpha=0.5, label='t-Threshold')
        plt.axhline(-thresh_t, color='red', linestyle='--')
        
        for start_s, end_s in significant_periods:
            plt.axvspan(start_s, end_s, color='yellow', alpha=0.3, label='Significant Window')

        plt.title(f'Cluster-based Permutation Test: {self.effect_name_dict[target_cols]} Effect\n{y_column}')
        plt.xlabel('Time (s)')
        plt.ylabel('t-statistic')
        
        # 凡例の重複除去
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()