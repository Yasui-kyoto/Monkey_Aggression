import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class DataCleaner:
    def __init__(self):
        pass
    
    

    def process_datetime_columns(self, df, year=2025):
        """
        Date列(20-Apr)とtime列(8:49:00)を統合し、
        日本で一般的な 'YYYY-MM-DD HH:MM:SS' 形式のdatetime列を作成する。
        """
        # 1. 2025-20-Apr 8:49:00 のような文字列を作成
        # Date列には '20-Apr' が入っていることを想定
        datetime_str = str(year) + '-' + df['Date'].astype(str) + ' ' + df['time'].astype(str)

        # 2. datetime型に変換
        # %d- %b は '20-Apr' の形式に対応
        df['datetime'] = pd.to_datetime(
            datetime_str, 
            format='%Y-%d-%b %H:%M:%S'
        )

        # 3. 元の列を削除
        df = df.drop(columns=['Date', 'time'])

        # 4. 列の並び替え（datetimeを一番左へ）
        cols = ['datetime'] + [c for c in df.columns if c != 'datetime']
        df = df[cols]

        return df
    


    def drop_unnecessary_columns(self, df):
        """
        指定された不要な列をデータフレームから削除する。
        """
        # 削除対象のリスト
        cols_to_drop = [
            "facehigh", 
            "facelow", 
            "facehigh.1", 
            "facelow.1", 
            "luminocity",
        ]

        # 実際にデータフレーム内に存在する列だけを抽出して削除
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols)

        return df



    def add_sampling_id(self, df):
        """
        データフレームに連続した記録セッションごとの sampling_id を付与する。
        """
        # 1. 時間の差分を計算 (単位: 秒)
        # 前の行との差分を秒数で取得
        time_diff = df['datetime'].diff().dt.total_seconds()

        # 2. セッション切り替わり条件の判定
        # 条件1: t0_flag が 1 である
        is_t0 = df['t0_flag'] == 1
        
        # 条件2: 前の行から個体名 (name) が変わった
        is_new_name = df['name'] != df['name'].shift(1)
        
        # 条件3: 前の行から10分 (600秒) 以上経過している
        is_time_gap = time_diff >= 600

        # 全ての条件を論理和 (OR) で統合
        # いずれかを満たせば新しいセッションの開始点 (True) となる
        new_session_trigger = is_t0 | is_new_name | is_time_gap
        
        # 最初の行は必ず新しいセッションとして開始させる
        new_session_trigger.iloc[0] = True 
        
        # --- 追加：t0_flag の更新 ---
        # 判定がTrueの行は 1.0 に、Falseの行は NaN に設定
        df['t0_flag'] = np.where(new_session_trigger, 1.0, np.nan)

        # 3. True (1) の累積和を計算することで、ユニークな ID を生成
        df['sampling_id'] = new_session_trigger.cumsum()
        
        # 4. 列の並び替え（sampling_idを一番左へ）
        other_cols = [c for c in df.columns if c not in ['sampling_id', 't0_flag']]
        cols = ['sampling_id', 't0_flag'] + other_cols
        df = df[cols]

        return df



    def add_behavior_analysis_cols(self, df):
        """
        sampling_id内での行動変化を分析し、フラグ列を追加する。
        """
        # SettingWithCopyWarningを避けるため、明示的にコピーを作成
        df = df.copy()

        # 1. 必要な列だけに絞って処理を行い、FutureWarningを回避
        # applyに全ての列を渡さず、'behavior'列のみを対象にします
        grouped_bh = df.groupby('sampling_id')['behavior']

        # --- A. groomer <-> groomee の変化があったか ---
        def check_role_swap(series):
            # unique()の結果からセットを作成
            unique_bh = set(series.unique())
            return ('grooming' in unique_bh) and ('groomed' in unique_bh)
        
        # transformを使えばmapの手間が省け、かつ安全です
        df['is_role_swapped'] = grouped_bh.transform(check_role_swap)

        # --- B. Sg (停滞) があったか ---
        def check_stagnant(series):
            return 'Sg' in series.values
            
        df['has_Sg'] = grouped_bh.transform(check_stagnant)

        # --- C. セッション内で最初のbehaviorが継続しているか ---
        def get_initial_continuation(series):
            first_bh = series.iloc[0]
            # 各行が最初のbehaviorと一致しているか
            matches_first = (series == first_bh)
            # 一度Falseになるとそれ以降すべてFalse
            return matches_first.cummin()

        df['is_initial_behavior'] = grouped_bh.transform(get_initial_continuation)

        return df



    def filter_initial_behavior_only(self, df):
        """
        解析のノイズとなる「最初の行以外」を削除する。
        """
        # is_initial_behavior が True の行だけを残す
        if 'is_initial_behavior' in df.columns:
            df = df[df['is_initial_behavior'] == True].copy()
            # フラグ列自体はもう不要なら削除
            df = df.drop(columns=['is_initial_behavior'])
        
        return df
    


    def add_temperature_delta(self, df):
        """
        sampling_idごとにループを回し、開始点を0秒/0度とした変化量を計算する。
        """
        # SettingWithCopyWarningを避けるためコピーを作成
        df = df.copy()
        
        # 1. 温度列を数値型に変換（前処理）
        df['facetemp'] = pd.to_numeric(df['facetemp'], errors='coerce')
        df['nosetemp'] = pd.to_numeric(df['nosetemp'], errors='coerce')
        
        # 結果を格納するリスト
        processed_chunks = []
    
        # 2. sampling_id ごとにグループ化して処理
        for sid, group in df.groupby('sampling_id'):
            # 時間順にソート（基準点を正しく取得するため）
            group = group.sort_values('datetime')
            
            # 基準となる「最初の行（t0）」の値を取得
            t0 = group['datetime'].iloc[0]
            f0 = group['facetemp'].iloc[0]
            n0 = group['nosetemp'].iloc[0]
            
            # --- 経過時間の計算（秒数に変換） ---
            # (現在時刻 - 開始時刻) の差分から、トータルの秒数を抽出
            group['delta_time'] = (group['datetime'] - t0).dt.total_seconds()
            
            # --- 温度変化の計算 ---
            group['delta_face'] = group['facetemp'] - f0
            group['delta_nose'] = group['nosetemp'] - n0
            
            processed_chunks.append(group)
    
        # 3. 全てのグループを一つに結合
        new_df = pd.concat(processed_chunks).reset_index(drop=True)
    
        # 4. 不要になった元の温度列を削除
        new_df = new_df.drop(columns=['facetemp', 'nosetemp'])
    
        # 5. 列の配置を調整
        current_cols = new_df.columns.tolist()
        if 'datetime' in current_cols:
            insert_idx = current_cols.index('datetime') + 1
            delta_cols = ['delta_time', 'delta_face', 'delta_nose']
            other_cols = [c for c in current_cols if c not in delta_cols]
            final_cols = other_cols[:insert_idx] + delta_cols + other_cols[insert_idx:]
            new_df = new_df[final_cols]
    
        return new_df
    
    
    
    def print_sample_counts(self, df):
        """
        有効な総サンプル数（sampling_idの種類数）と、
        各behaviorごとのサンプル数を表示する。
        """
        # 1. 有効な総サンプル数（sampling_idのユニーク数）
        total_samples = df['sampling_id'].nunique()
        
        print("========== サンプル数集計 ==========")
        print(f"有効な総サンプル数: {total_samples}")
        print("------------------------------------")
        
        # 2. behaviorごとのサンプル数
        # behaviorごとにグループ化し、それぞれのグループ内にあるsampling_idのユニーク数をカウント
        behavior_counts = df.groupby('behavior')['sampling_id'].nunique()
        
        print("各behaviorごとのサンプル数:")
        for behavior, count in behavior_counts.items():
            print(f"  - {behavior}: {count}")
        print("====================================")
    
    
    
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
        
        

    def plot_sample_with_shade_transition(self, df, target_id=19, y_column='delta_nose'):
        """
        特定のサンプルの温度変化を折れ線で表示し、背景色でshadeの切り替わりを可視化する。
        """
        # 対象のサンプリングIDのみを抽出
        sample_df = df[df['sampling_id'] == target_id].sort_values('delta_time')

        if sample_df.empty:
            print(f"Sampling ID {target_id} はデータ内に存在しません。")
            return

        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        # 1. 背景にshadeのエリアを色付け
        # shadeが 'yes' (日陰) の区間を薄い青、 'no' (日向) の区間を薄いオレンジなどで塗る
        times = sample_df['delta_time'].values
        shades = sample_df['shade'].values

        # 区間ごとに背景色を塗るループ
        for i in range(len(times) - 1):
            color = 'skyblue' if shades[i] == 'yes' else 'orange'
            ax.axvspan(times[i], times[i+1], color=color, alpha=0.2)

        # 2. 温度変化を折れ線グラフでプロット
        plt.plot(sample_df['delta_time'], sample_df[y_column], 
                 marker='o', markersize=4, color='black', linewidth=1.5, label=y_column)

        # 凡例用のダミー（背景色の説明用）
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='skyblue', lw=4, alpha=0.3),
                        Line2D([0], [0], color='orange', lw=4, alpha=0.3),
                        Line2D([0], [0], color='black', lw=1.5, marker='o')]
        ax.legend(custom_lines, ['Shade: yes (Shadow)', 'Shade: no (Sun)', y_column], loc='upper left')

        position_dict = {'delta_face': 'Face', 'delta_nose': 'Nose'}
        plt.title(f'Sample ID {target_id}: {position_dict.get(y_column, y_column)} Temp Change vs Shade Transition')
        plt.xlabel('Time from start (s)')
        plt.ylabel('Temperature change (°C)')
        plt.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.show()
        
        

    def filter_by_shade_consistency(self, df, threshold=0.9):
        """
        sampling_idごとにshadeの一貫性をチェックし、
        支配的なshade状態がthreshold（デフォルト90%）未満のサンプルを除外する。
        """
        # フィルタリング前のサンプル数を記録
        before_df = df.copy()
        
        # sampling_idごとに、最も頻繁に現れるshadeの割合を計算
        def check_consistency(group):
            # 各shadeの状態（yes/no）の出現回数をカウント
            counts = group['shade'].value_counts(normalize=True)
            # 最大の割合（最も支配的な状態の割合）を返す
            return counts.max() >= threshold

        # 条件を満たすsampling_idを特定
        consistent_ids = df.groupby('sampling_id').apply(
            lambda x: check_consistency(x), include_groups=False
        )
        valid_ids = consistent_ids[consistent_ids].index
        
        # フィルタリングの実行
        filtered_df = df[df['sampling_id'].isin(valid_ids)].copy()
        
        return filtered_df
    


    def plot_behavior_shade_comparison(self, df, target_behavior='BL', y_column='delta_face', threshold=0.9):
        """
        特定のbehaviorにおいて、常に日向(no)のサンプルと常に日陰(yes)のサンプルを比較プロットする。
        """
        # 1. 特定のbehaviorで絞り込み
        b_df = df[df['behavior'] == target_behavior].copy()
        
        if b_df.empty:
            print(f"Behavior '{target_behavior}' のデータが存在しません。")
            return
    
        # 2. 各サンプルの支配的なshadeを判定
        def get_dominant_shade(group):
            counts = group['shade'].value_counts(normalize=True)
            if counts.max() >= threshold:
                return counts.idxmax() # 'yes' か 'no' を返す
            else:
                return 'mixed' # 混合サンプル
    
        # 各IDの環境ラベルを作成
        shade_labels = b_df.groupby('sampling_id').apply(
            lambda x: get_dominant_shade(x), include_groups=False
        ).to_dict()
        
        # 元のデータフレームにラベルをマップ
        b_df['shade_condition'] = b_df['sampling_id'].map(shade_labels)
    
        # 3. 'mixed' を除外し、比較用のプロットを作成
        plot_df = b_df[b_df['shade_condition'] != 'mixed']
        
        plt.figure(figsize=(10, 6))
        
        # 日向=オレンジ, 日陰=青系の色使い
        shade_palette = {'no': '#ff7c00', 'yes': '#023eff'}
        shade_names = {'no': 'Always Sun (no)', 'yes': 'Always Shadow (yes)'}
    
        sns.scatterplot(
            data=plot_df,
            x='delta_time',
            y=y_column,
            hue='shade_condition',
            palette=shade_palette,
            alpha=0.6
        )
    
        # 統計情報の取得（サンプル数の確認）
        counts = plot_df.groupby('shade_condition')['sampling_id'].nunique()
        sun_n = counts.get('no', 0)
        sha_n = counts.get('yes', 0)
    
        position_dict = {'delta_face': 'Face', 'delta_nose': 'Nose'}
        plt.title(f'Environmental Impact on {target_behavior}\n'
                  f'Sun (n={sun_n}) vs Shadow (n={sha_n})')
        plt.xlabel('Time from the starting point (s)')
        plt.ylabel(f'{position_dict.get(y_column, y_column)} Temperature change (°C)')
        
        # 凡例のラベルを分かりやすく変更
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, [shade_names.get(l, l) for l in labels], 
                   title='Environment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()