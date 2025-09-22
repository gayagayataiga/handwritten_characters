import torch

# ファイルのパス
file_path = 'gemi.pth'

try:
    # モデルのstate_dictを読み込む
    # map_location='cpu'を指定すると、GPUがない環境でもエラーなく読み込めます
    state_dict = torch.load(file_path, map_location='cpu')

    # 読み込んだデータの型を確認
    print(f"読み込んだデータの型: {type(state_dict)}\n")

    # state_dictが辞書の場合、キー（層の名前）を一覧表示
    if isinstance(state_dict, dict):
        print("モデルの層 (Keys):")
        for key in state_dict.keys():
            print(key)
    else:
        print("辞書形式ではありませんでした。読み込んだ内容を直接表示します。")
        print(state_dict)

except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("このファイルはstate_dictではないか、破損している可能性があります。")