import streamlit as st
from PIL import Image
import os
from fashion_detection_gui import predict

st.title("Tìm kiếm ảnh liên quan")

uploaded_file = st.file_uploader("Tải lên ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    st.write("")
    
    # Sử dụng mô hình để tìm ảnh liên quan
    # similar_images = predict(uploaded_file, "dataset")

    # st.write(f"Hiển thị {len(similar_images)} ảnh tương tự:")

    # cols = st.columns(4)
    # for idx, img_info in enumerate(similar_images):
    #     img = Image.open(img_info["image_path"])
    #     with cols[idx % 4]:
    #         st.image(img, use_column_width=True)
    #         st.caption(img_info["info"])
