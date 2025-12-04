import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Brain Lesion Dashboard", layout="wide")

# ---------------------------- UI -----------------------------
st.title("🧠 Brain MRI Lesion Detection Dashboard")

# Sidebar
uploaded_file = st.sidebar.file_uploader("上传 MRI 图像", type=["png", "jpg", "jpeg"])
threshold = st.sidebar.slider("阈值参数", 0, 255, 120)
kernel_size = st.sidebar.slider("形态学核大小", 1, 15, 5)

# ----------------------- Image Pipeline -----------------------
def detect_lesion(img):
    # 去噪
    denoised = cv2.medianBlur(img, 5)

    # 阈值
    _, binary = cv2.threshold(denoised, threshold, 255, cv2.THRESH_BINARY)

    # 形态学
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 叠加伪彩色
    overlay = cv2.applyColorMap(morph, cv2.COLORMAP_JET)

    return denoised, binary, morph, overlay

# ---------------------- Main Layout --------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("病例检测结果展示")

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        denoised, binary, morph, overlay = detect_lesion(img)

        # 两列大图
        c1, c2 = st.columns(2)
        c1.image(img, caption="原始 MRI", use_column_width=True)
        c2.image(overlay, caption="病灶叠加图", use_column_width=True)

        st.subheader("处理流程")
        st.image([denoised, binary, morph], caption=["去噪", "阈值分割", "形态学"], width=200)

with col2:
    st.subheader("统计分析")
    
    # 模拟多例数据（你之后可以替换为真实数据）
    lesion_areas = np.random.randint(1000, 9000, size=50)
    
    st.metric("平均病灶面积", f"{np.mean(lesion_areas):.0f} px²")
    st.metric("最大病灶面积", f"{np.max(lesion_areas):.0f} px²")
    st.metric("阳性比例", "28%")

    st.bar_chart(lesion_areas)

    # 阳性 vs 阴性示例
    pos_neg = pd.DataFrame({"label": ["Positive", "Negative"], 
                            "count": [14, 36]})
    st.bar_chart(pos_neg.set_index("label"))

# ---------------------- Bottom Gallery -------------------------
st.subheader("批量检测缩略图展示（示例）")

gallery_cols = st.columns(6)
for i, col in enumerate(gallery_cols):
    col.image(np.random.randint(0, 255, (240, 240)), caption=f"Case #{i+1}", width=120)
