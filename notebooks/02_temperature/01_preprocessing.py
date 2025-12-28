import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    # 現在のノートブックのパスを取得
    notebook_dir = Path().resolve() 

    print(notebook_dir)

    # プロジェクトルートディレクトリ（notebooksディレクトリの親ディレクトリ）を取得
    # プロジェクトルートは2階層上です
    project_root = notebook_dir.parent.parent

    # プロジェクトルートをPythonの検索パスに追加
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dataの読み込み
    """)
    return


if __name__ == "__main__":
    app.run()
