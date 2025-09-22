import torch

file_path = 'gemi.pth'
state_dict = torch.load(file_path, map_location='cpu')

if isinstance(state_dict, dict):
    print("各層のパラメータの形状 (Tensor Shape):")
    for key, value in state_dict.items():
        # valueはパラメータのテンソル
        print(f"- {key}: {value.shape}")
else:
    print("辞書形式ではないため、詳細な解析は困難です。")