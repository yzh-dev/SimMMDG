import os

# 定义下载 URL 和 目标路径
url = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/test/P01/P01_11.tar"
output_path = r"C:\Users\Administrator\Downloads\EPIC_KITCHENS2\frames_rgb_flow\flow\test\D2"

# 确保目标目录存在
os.makedirs(output_path, exist_ok=True)

# 构造 wget 命令
output_file = os.path.join(output_path, "P01_11.tar")
wget_command = f'wget -O "{output_file}" "{url}"'

# 执行 wget 命令
os.system(wget_command)

print(f"Download completed: {output_file}")
