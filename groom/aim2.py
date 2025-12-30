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