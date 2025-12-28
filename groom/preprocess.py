import numpy as np
import pandas as pd



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
            "luminocity"
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

        # 3. True (1) の累積和を計算することで、ユニークな ID を生成
        df['sampling_id'] = new_session_trigger.cumsum()

        return df