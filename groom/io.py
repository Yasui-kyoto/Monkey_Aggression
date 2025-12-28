import pandas as pd
import os
import pickle



def save_to_parquet(df: pd.DataFrame, filename: str):
    """ DataFrameをParquetファイルとして保存します。 """
    # to_parquet() を使用。インデックスは自動的に保存されます。
    df.to_parquet(filename)
    print(f"--- 保存完了 ---")
    print(f"'{filename}' として保存されました。ファイルサイズ: {os.path.getsize(filename) / 1024:.2f} KB")



def load_from_parquet(filename: str) -> pd.DataFrame:
    """ ParquetファイルからDataFrameを復元します。 """
    # read_parquet() を使用。インデックスとデータ型はロスレスで復元されます。
    loaded_df = pd.read_parquet(filename)
    print(f"--- 復元完了 ---")
    print("復元されたDataFrameのshape:", loaded_df.shape)
    return loaded_df