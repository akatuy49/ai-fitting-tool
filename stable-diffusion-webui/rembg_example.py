from rembg import remove
from PIL import Image

input_image = Image.open("tshirt.jpg")         # 元画像（背景つき）
output_image = remove(input_image)             # 背景を削除
output_image.save("tshirt_cutout.png")         # 背景透明のPNGとして保存
