import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats



class GroomAim1:
    def __init__(self):
        self.position_dict = {
            'delta_face': 'Face',
            'corrected_delta_face': 'Face (Corrected)',
            'delta_nose': 'Nose',
            'corrected_delta_nose': 'Nose (Corrected)',
            'nose-face': 'Nose - Face',
            'corrected_nose-face': 'Corrected Nose - Face'
        }
        pass
    
    
    
    def plot_behavior_scatter(self, df, y_column, x_column='delta_time', hue_column='behavior'):
        """
        指定されたデータフレームから散布図を作成する関数。
        """
        plt.figure(figsize=(10, 6))
        
        # behaviorごとの色を固定する辞書
        color_palette = {'BL': '#023eff', 'grooming': '#ff7c00', 'groomed': '#1ac938'}
    
        # --- 修正ポイント1: 凡例ラベルに n= を追加 ---
        # 各カテゴリーの出現回数をカウント
        counts = df[hue_column].value_counts()
        
        # 元のデータフレームをコピーして、凡例用のラベル書き換えた一時的なカラムを作成
        df_plot = df.copy()
        df_plot['legend_label'] = df_plot[hue_column].apply(lambda x: f"{x} (n={counts[x]})")
        
        # カラーパレットも新しいラベルに対応させる
        new_palette = {f"{k} (n={counts[k]})": v for k, v in color_palette.items() if k in counts}
    
        # 散布図の描画
        sns.scatterplot(
            data=df_plot, 
            x=x_column, 
            y=y_column, 
            hue='legend_label',  # 新しく作ったラベルを使用
            palette=new_palette,
            alpha=0.7
        )
    
        
        # --- 修正ポイント2: タイトルに総数を表示 ---
        total_n = len(df)
        plt.title(f'Time Series Analysis: Scatter plot of {self.position_dict[y_column]} Change (Total n={total_n})')
        
        plt.xlabel('Time from the starting point (s)')
        plt.ylabel('Temperature change (°C)')
        
        # 凡例の設定
        plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    

    def plot_highlight_behavior(self, df, target_behavior, y_column, x_column='delta_time'):
        """
        特定のbehaviorだけを色付けし、それ以外を灰色で表示する関数。
        """
        plt.figure(figsize=(10, 6))
        
        # behaviorごとの色を固定する辞書
        color_palette = {'BL': '#023eff', 'grooming': '#ff7c00', 'groomed': '#1ac938'}
        target_color = color_palette.get(target_behavior, 'red')

        # データを分離
        other_df = df[df['behavior'] != target_behavior]
        target_df = df[df['behavior'] == target_behavior]
        
        # --- 修正ポイント1: 各サンプル数を計算 ---
        n_others = len(other_df)
        n_target = len(target_df)
        total_n = len(df)

        # 1. ターゲット以外のデータを灰色でプロット
        sns.scatterplot(
            data=other_df,
            x=x_column,
            y=y_column,
            color='gray',
            alpha=0.25,
            label=f'Others (n={n_others})'  # ラベルに n= を追加
        )

        # 2. ターゲットのデータだけを色付きで重ねてプロット
        sns.scatterplot(
            data=target_df,
            x=x_column,
            y=y_column,
            color=target_color,
            alpha=0.8,
            label=f'{target_behavior} (n={n_target})'  # ラベルに n= を追加
        )
        
        # --- 修正ポイント2: タイトルに総数を表示 ---
        plt.title(f'Highlight: {target_behavior} ({self.position_dict[y_column]}) | Total n={total_n}')
        
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
            # そのサンプルの「開始時刻」を取得しておく
            start_datetime = group['datetime'].min()
            
            new_index = np.arange(0, time_limit + 1)
            temp_group = group.drop_duplicates(subset='delta_time').set_index('delta_time')

            # 器の作成と数値補間
            resampled = temp_group.reindex(new_index)
            
            # --- 修正ポイント ---
            # 1. 数値列のみ線形補間
            numeric_cols = resampled.select_dtypes(include=[np.number]).columns
            resampled[numeric_cols] = resampled[numeric_cols].interpolate(method='linear')
            
            # 2. 非数値列も含めて前後埋め
            # future.no_silent_downcasting を設定するか、明示的にキャスト
            resampled = resampled.ffill().bfill()
            resampled = resampled.infer_objects(copy=False)
            # --------------------
            
            # datetime列（pd.datetime型） を再計算する ---
            # 開始時刻に delta_time (index) を秒として加算する
            resampled['datetime'] = start_datetime + pd.to_timedelta(resampled.index, unit='s')
            # ---------------------------------------
            
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
        


    def run_cluster_permutation_test(self, df_1s, target_behaviors=['grooming', 'groomed'], y_column='delta_nose', n_permutations=1000, p_threshold=0.05):
        """
        有意なクラスターの時間帯を特定して表示する機能を追加したクラスターベース置換検定。
        """
        # behavior
        behavior_a = target_behaviors[0]
        behavior_b = target_behaviors[1]
        
        # 1. データの準備
        data_a = df_1s[df_1s['behavior'] == behavior_a]
        data_b = df_1s[df_1s['behavior'] == behavior_b]
        times = sorted(df_1s['delta_time'].unique())
        
        def create_matrix(sub_df):
            matrix = sub_df.pivot(index='sampling_id', columns='delta_time', values=y_column)
            return matrix.dropna().values

        matrix_a = create_matrix(data_a)
        matrix_b = create_matrix(data_b)
        
        if len(matrix_a) < 3 or len(matrix_b) < 3:
            print("サンプル数が少なすぎるため、検定をスキップします。")
            return None

        # 2. 実データの t値計算とクラスター特定
        t_obs, p_obs = stats.ttest_ind(matrix_a, matrix_b, axis=0)
        df_degree = len(matrix_a) + len(matrix_b) - 2
        thresh_t = stats.t.ppf(1 - p_threshold / 2, df_degree)
        
        # クラスターの情報を保持する関数（開始・終了時間も記録）
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
        # 各クラスターのt値合計の絶対値を計算
        obs_cluster_stats = [np.abs(np.sum(t_obs[c])) for c in obs_clusters]
        obs_max_cluster = np.max(obs_cluster_stats) if obs_cluster_stats else 0

        # 3. 置換検定（帰無分布の作成）
        print(f"Running {n_permutations} permutations...")
        combined_matrix = np.vstack([matrix_a, matrix_b])
        n_a = len(matrix_a)
        null_max_clusters = []

        for _ in range(n_permutations):
            indices = np.random.permutation(len(combined_matrix))
            t_rand, _ = stats.ttest_ind(combined_matrix[indices[:n_a]], combined_matrix[indices[n_a:]], axis=0)
            
            # シャッフル時の最大クラスター統計量を記録
            rand_clusters = find_clusters(t_rand, thresh_t)
            rand_stats = [np.abs(np.sum(t_rand[c])) for c in rand_clusters]
            null_max_clusters.append(np.max(rand_stats) if rand_stats else 0)

        # 4. 各クラスターごとにP値を計算 ---
        print(f"\n--- Cluster-based Permutation Result: {behavior_a} vs {behavior_b} ---")
        
        # 帰無分布（null_max_clusters）はステップ4で作成したものを使用
        null_dist = np.array(null_max_clusters)
        
        significant_periods = []
        
        if len(obs_clusters) == 0:
            print("有意なクラスターは検出されませんでした。")
        else:
            for i, c_indices in enumerate(obs_clusters):
                # この個別のクラスターの合計値が、シャッフル時の「最大値分布」の中で何％の位置にあるか
                c_stat = obs_cluster_stats[i]
                c_p_value = np.sum(null_dist >= c_stat) / n_permutations

                start_s = times[c_indices[0]]
                end_s = times[c_indices[-1]]

                print(f"Cluster {i+1}: {start_s}s - {end_s}s | Stat: {c_stat:.2f} | p = {c_p_value:.4f}")

                if c_p_value < p_threshold:
                    print(f"  => ★ 有意 (p < {p_threshold})")
                    significant_periods.append((start_s, end_s))
                else:
                    print(f"  => 有意差なし")
                
        print("---------------------------------------------------------")

        # 5. 可視化
        plt.figure(figsize=(12, 5))
        plt.plot(times, t_obs, label='Observed t-value', color='black', alpha=0.7)
        plt.axhline(thresh_t, color='red', linestyle='--', alpha=0.5, label='Threshold')
        plt.axhline(-thresh_t, color='red', linestyle='--', alpha=0.5)
        
        for start_s, end_s in significant_periods:
            plt.axvspan(start_s, end_s, color='yellow', alpha=0.3, label='Significant Window')

        plt.title(f'Cluster-based Permutation Test: {behavior_a} vs {behavior_b} ({self.position_dict[y_column]})')
        plt.xlabel('Time (s)')
        plt.ylabel('t-statistic')
        # 重複する凡例を避ける
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        
        
    def _get_bl_trend(self, interpolated_df, y_column='delta_nose'):
        """
        BL（基準）行動の各秒における平均値を抽出する関数。
        """
        bl_data = interpolated_df[interpolated_df['behavior'] == 'BL']
        
        # 秒ごとの平均値を計算
        bl_trend = bl_data.groupby('delta_time')[y_column].mean()
        
        return bl_trend
    
    

    def apply_baseline_correction(self, interpolated_df, y_columns=['delta_nose', 'delta_face']):
        """
        各群の値を、同じ時刻のBL平均値で差し引いて補正する関数。
        新しい列 'corrected_delta_nose' などを作成したDataFrameを返す。
        """
        df_corr = interpolated_df.copy()
        
        for col in y_columns:
            # BLトレンドの取得
            bl_trend = self._get_bl_trend(df_corr, y_column=col)
            
            # 各行のdelta_timeに対応するBL平均値をマッピングして差し引く
            # mapを使うことで高速に処理できます
            df_corr[f'corrected_{col}'] = df_corr[col] - df_corr['delta_time'].map(bl_trend)
            
        print(f"補正が完了しました。新しく追加された列: {[f'corrected_{c}' for c in y_columns]}")
        return df_corr
    
    
    
    def calculate_nose_face_difference(self, df, use_corrected=False):
        """
        noseの温度変化からfaceの温度変化を引いた値を算出する。
        
        Parameters:
        -----------
        df : pd.DataFrame
            入力データフレーム
        use_corrected : bool, default False
            Trueの場合、'corrected_delta_nose' と 'corrected_delta_face' を使用する。
            Falseの場合、'delta_nose' と 'delta_face' を使用する。
        """
        df_res = df.copy()
        
        if use_corrected:
            # 補正済み列を使用する場合
            target_nose = 'corrected_delta_nose'
            target_face = 'corrected_delta_face'
            new_col_name = 'corrected_nose-face'
        else:
            # 通常のdelta列を使用する場合
            target_nose = 'delta_nose'
            target_face = 'delta_face'
            new_col_name = 'nose-face'
            
        # 必要な列が存在するかチェック
        if target_nose in df_res.columns and target_face in df_res.columns:
            df_res[new_col_name] = df_res[target_nose] - df_res[target_face]
            print(f"計算が完了しました。新しい列: '{new_col_name}' を作成しました。")
        else:
            print(f"エラー: 必要な列 ({target_nose}, {target_face}) が見つかりません。")
            
        return df_res
    
    
    
    def run_one_sample_cluster_test(self, df_1s, behavior='grooming', y_column='corrected_nose-face', n_permutations=1000, p_threshold=0.05):
        """
        指定した行動の y_column が y=0 と有意に異なる時間帯を特定する1標本クラスターベース置換検定。
        """
        # 1. データの準備
        data_sub = df_1s[df_1s['behavior'] == behavior]
        times = sorted(df_1s['delta_time'].unique())
        
        def create_matrix(sub_df):
            # Rows: sampling_id, Cols: delta_time
            matrix = sub_df.pivot(index='sampling_id', columns='delta_time', values=y_column)
            return matrix.dropna().values

        matrix_obs = create_matrix(data_sub)
        
        if len(matrix_obs) < 3:
            print(f"サンプル数が少なすぎるため（n={len(matrix_obs)}）、検定をスキップします。")
            return None

        # 2. 実データの 1標本t値計算 (vs 0) とクラスター特定
        # stats.ttest_1samp はデフォルトで 0 と比較します
        t_obs, p_obs = stats.ttest_1samp(matrix_obs, 0, axis=0)
        df_degree = len(matrix_obs) - 1
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
        
        # 3. 置換検定（符号反転による帰無分布の作成）
        print(f"Running {n_permutations} permutations (sign flipping)...")
        n_samples = matrix_obs.shape[0]
        null_max_clusters = []

        for _ in range(n_permutations):
            # 各個体ごとに +1 または -1 をランダムに生成して掛ける
            signs = np.random.choice([-1, 1], size=(n_samples, 1))
            matrix_shuffled = matrix_obs * signs
            
            t_rand, _ = stats.ttest_1samp(matrix_shuffled, 0, axis=0)
            
            rand_clusters = find_clusters(t_rand, thresh_t)
            rand_stats = [np.abs(np.sum(t_rand[c])) for c in rand_clusters]
            null_max_clusters.append(np.max(rand_stats) if rand_stats else 0)

        # 4. 各クラスターごとにP値を計算
        print(f"\n--- One-sample Cluster Test Result: {behavior} vs 0 ---")
        print(f"Target variable: {self.position_dict.get(y_column, y_column)}")
        
        null_dist = np.array(null_max_clusters)
        significant_periods = []
        
        if len(obs_clusters) == 0:
            print("有意なクラスターは検出されませんでした。")
        else:
            for i, c_indices in enumerate(obs_clusters):
                c_stat = obs_cluster_stats[i]
                c_p_value = np.sum(null_dist >= c_stat) / n_permutations
                
                start_s = times[c_indices[0]]
                end_s = times[c_indices[-1]]
                
                print(f"Cluster {i+1}: {start_s}s - {end_s}s | Stat: {c_stat:.2f} | p = {c_p_value:.4f}")
                
                if c_p_value < p_threshold:
                    print(f"  => ★ 有意 (p < {p_threshold})")
                    significant_periods.append((start_s, end_s))
                else:
                    print(f"  => 有意差なし")
        print("---------------------------------------------------------")

        # 5. 可視化
        plt.figure(figsize=(12, 5))
        plt.plot(times, t_obs, label='Observed t-value (vs 0)', color='black', alpha=0.7)
        plt.axhline(thresh_t, color='red', linestyle='--', alpha=0.5, label='Threshold')
        plt.axhline(-thresh_t, color='red', linestyle='--', alpha=0.5)
        plt.axhline(0, color='gray', linewidth=0.8) # 0ライン
        
        for start_s, end_s in significant_periods:
            plt.axvspan(start_s, end_s, color='yellow', alpha=0.3, label='Significant Window')

        plt.title(f'One-sample Cluster Test: {behavior}\n({self.position_dict.get(y_column, y_column)})')
        plt.xlabel('Time (s)')
        plt.ylabel('t-statistic')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        
 
        
class GroomAim1LinearTests:
    def __init__(self):
        self.position_dict = {
            'delta_face': 'Face',
            'corrected_delta_face': 'Face (Corrected)',
            'delta_nose': 'Nose',
            'corrected_delta_nose': 'Nose (Corrected)'
        }
        pass
    
    
    
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

        plt.title(f'Trend Comparison: {behavior_a} vs {behavior_b}\n(Linear Regression & 95% CI)')
        plt.xlabel('Time from the starting point (s)')
        plt.ylabel(f'{self.position_dict.get(y_column, y_column)} Temperature change (°C)')
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

        plt.title(f'Final {self.position_dict.get(y_column, y_column)} Temperature Change\n(Last point of each sample)')
        plt.ylabel(f'{self.position_dict.get(y_column, y_column)} Delta (°C)')
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()