import streamlit as st
from PIL import Image
from io import BytesIO
from predict import pred
from utils import RGB2GRAY
import cv2

st.set_page_config(layout="wide", page_title="Grayscale Image Colorizer")

st.write("## Colorize grayscale image")
st.write(
    ":dog: Try uploading a grayscale image to colorize it :grin:"
)

st.sidebar.write("## Upload and download :gear:")

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    col1.write("Original Image :camera:")
    img = Image.open(my_upload)
    w, h = img.size
    imgGray = img.convert('L')
    col1.image(imgGray)
    pred_img = pred(my_upload)
    col2.write("Colorized Image :wrench:")
    col2.image(cv2.resize(pred_img, (w,h), interpolation = cv2.INTER_AREA))

st.sidebar.markdown("\n")