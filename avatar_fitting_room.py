import os
import streamlit as st
from PIL import Image
import requests
import base64
from io import BytesIO

AVATAR_FOLDER = "avatars"
DEFAULT_CLOTHING = "tshirt_cutout.png"
API_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"

st.title("🌟 アバターに服を着せるツール (ControlNet Canny版)")

avatar_files = [f for f in os.listdir(AVATAR_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not avatar_files:
    st.warning("avatarsフォルダに画像がありません。")
    st.stop()

st.subheader("👤 アバターを選択")
selected_avatar = st.selectbox("アバター画像を選んでください", avatar_files)
avatar_path = os.path.join(AVATAR_FOLDER, selected_avatar)
st.image(Image.open(avatar_path), caption="選択中のアバター", width=256)

st.subheader("👕 着せたい服画像をアップロード(PNG)")
uploaded_file = st.file_uploader("服画像(背景透過のPNG)を選択", type=["png"])

if uploaded_file is not None:
    clothing_img = Image.open(uploaded_file).convert("RGB").resize((512, 512))
else:
    clothing_img = Image.open(DEFAULT_CLOTHING).convert("RGB").resize((512, 512))
    st.caption("*デフォルトのTシャツ画像が使用されます*")

buffer = BytesIO()
clothing_img.save(buffer, format="PNG")
clothing_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
st.image(clothing_img, caption="使用する服画像", width=256)

st.subheader("✨ アバターに服を着せる")
if st.button("生成 (Generate)"):
    st.info("ControlNetで画像生成中...")

    payload = {
        "prompt": "a woman wearing a red T-shirt, full body, photo, plain background",
        "negative_prompt": "blurry, watermark, logo, extra limbs",
        "sampler_name": "DPM++ 2M",
        "steps": 20,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": clothing_b64,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny [d14c016b]",
                        "weight": 1.0,
                        "resize_mode": 1,         # ←int型
                        "control_mode": 0,        # ←int型
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "starting_control_step": 0,
                        "ending_control_step": 1,
                        "pixel_perfect": True
                    }
                ]
            }
        }
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        st.json(result)

        if "images" in result:
            img_data = base64.b64decode(result["images"][0])
            with open("result.png", "wb") as f:
                f.write(img_data)
            st.image("result.png", caption="合成結果")
            st.success("服を着せた画像が生成されました！")
        else:
            st.error("画像生成に失敗しました。レスポンスに 'images' が含まれていません。")
    except Exception as e:
        st.error(f"画像生成エラー: {e}")
