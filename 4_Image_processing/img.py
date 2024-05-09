from image_processing import Chapter03
from image_processing import Chapter04
from image_processing import Chapter05
from image_processing import Chapter09
import streamlit as st
from PIL import Image
import cv2
import numpy as np
def app():
    st.title("Xử lý ảnh")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        option =  st.selectbox("Function", ("None", "Negative", 'Logarit', "Power", "Piecewise Linear", "Histogram", "Histogram Equalization",
                                    "Local Histogram", "Histogram Statistics", "MyBox Filter", "Box Filter", "Threshold", "Median Filter", "Sharpen", "Gradient",
                                    "Spectrum", "Frequency Filter", "Remove Moire",
                                    "Create Motion Noise", "Denoise Motion",
                                    "Connected Component", "Count Rice"))
        if option == "None":
            st.write("Vui lòng chọn option!")
        else:
            st.write("")
        uploaded_file = st.file_uploader("Upload Image")
        if uploaded_file is not None:
            st.write("Kích thước của file ảnh là: ", uploaded_file.size, "Bytes")
            st.write("Tên của ảnh: ", uploaded_file.name)
            st.write("---------------------------------------------------------------")
        if option == "Negative":
            st.write("A negative of an image is an image where its lightest areas appear as darkest and the darkest areas appear as lightest.")
            st.write("The appearance change from lightest to darkest and darkest to lightest is basically done in gray scale image and refers to the change of pixel intensity values from highest to lowest and lowest to highest.")
        elif option == "Logarit":
            st.write("The dynamic range of an image can be compressed by replacing each pixel value with its logarithm. This has the effect that low intensity pixel values are enhanced. ")
            st.write("Applying a pixel logarithm operator to an image can be useful in applications where the dynamic range may too large to be displayed on a screen (or to be recorded on a film in the first place).")
        elif option == "Power":
            st.write("The exponential and `raise to power' operators are two anamorphosis operators which can be applied to grayscale images. Like the logarithmic transform, they are used to change the dynamic range of an image. ")
            st.write("However, in contrast to the logarithmic operator, they enhance high intensity pixel values.")
        elif option == "Piecewise Linear":
            st.write("Piece-wise Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method.")
            st.write("It is used for manipulation of an image so that the result is more suitable than the original for a specific application.")
        elif option == "Histogram":
            st.write("Histograms Introduction. In digital image processing, the histogram is used for graphical representation of a digital image.")
            st.write("A graph is a plot by the number of pixels for each tonal value. Nowadays, image histogram is present in digital cameras. Photographers use them to see the distribution of tones captured.")
        elif option == "Histogram Equalization":
            st.write("Histogram Equalization is a computer image processing technique used to improve contrast in images ")
            st.write("It accomplishes this by effectively spreading out the most frequent intensity values, i.e. stretching out the intensity range of the image.")
        elif option == "Local Histogram":
            st.write("In digital image processing, the histogram is used for graphical representation of a digital image. A graph is a plot by the number of pixels for each tonal value. ")
            st.write("Nowadays, image histogram is present in digital cameras. Photographers use them to see the distribution of tones captured.")
        elif option == "Histogram Statistics":
            st.write("An image histogram is a type of histogram that acts as a graphical representation of the tonal distribution in a digital image.")
            st.write("In an image processing context, the histogram of an image normally refers to a histogram of the pixel intensity values.")
        elif option == "MyBox Filter":
            pass
        elif option == "Box Filter":
            pass
        elif option == "Threshold":
            pass
        elif option == "Smoothing":
            st.write("Smoothing is used to reduce noise or to produce a less pixelated image. Most smoothing methods are based on low-pass filters, ")
            st.write("But you can also smooth an image using an average or median value of a group of pixels (a kernel) that moves through the image.")
        elif option == "Sharpen":
            st.write("Image sharpening is an effect applied to digital images to give them a sharper appearance. ")
            st.write("Sharpening enhances the definition of edges in an image. The dull images are those which are poor at the edges. There is not much difference in background and edges.")
        elif option == "Gradient":
            st.write("An image gradient is a directional change in the intensity or color in an image. The gradient of the image is one of the fundamental building blocks in image processing")
            st.write("For example, the Canny edge detector uses image gradient for edge detection.")
        elif option == "Median Filter":
            st.write("The median filter is the filtering technique used for noise removal from images and signals.")
            st.write("Median filter is very crucial in the image processing field as it is well known for the preservation of edges during noise removal.")
        elif option == "Spectrum":
            pass
        elif option == "Frequency Filter":
            pass
        elif option == "Remove Moire":
            pass
        elif option == "Create Motion filter":
            pass
        elif option == "Create Motion Noise":
            pass
        elif option == "Denoise Motion":
            pass
        elif option == "Connected Component":
            pass
        elif option == "Count Rice":
            pass
        else:
            st.write("")

    with col2:
        global imgin
        if uploaded_file is not None:
            temp_image = Image.open(uploaded_file)
            image_path = 'image_processing/input/' + uploaded_file.name
            temp_image.save(image_path)
            image = Image.open(image_path)
            st.image(image, caption='Input', use_column_width=True)
            img_array = np.array(image)
            cv2.imwrite('image_processing/output/out.jpg', cv2.cvtColor(img_array, cv2.IMREAD_GRAYSCALE))
            imgin = cv2.imread('image_processing/output/out.jpg', cv2.IMREAD_GRAYSCALE)
        else:
            image_path = 'image_processing/data/background.jpg'
            image = Image.open(image_path)
            imgin = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            st.image(image, caption='Output', use_column_width=True)
    
        if option == "Negative":
            imgout = Chapter03.Negative(imgin)
        elif option == "Logarit":
            imgout = Chapter03.Logarit(imgin)
        elif option == "Power":
            imgout = Chapter03.Power(imgin)
        elif option == "Piecewise Linear":
            imgout = Chapter03.PiecewiseLinear(imgin)
        elif option == "Histogram":
            imgout = Chapter03.Histogram(imgin)
        elif option == "Histogram Equalization":
            imgout = Chapter03.HistEqual(imgin)
        elif option == "Local Histogram":
            imgout = Chapter03.LocalHist(imgin)
        elif option == "Histogram Statistics":
            imgout = Chapter03.HistStat(imgin)
        elif option == "MyBox Filter":
            imgout = Chapter03.MyBoxFilter(imgin)
        elif option == "Box Filter":
            imgout = Chapter03.BoxFilter(imgin)
        elif option == "Threshold":
            imgout = Chapter03.Threshold(imgin)
        elif option == "Sharpen":
            imgout = Chapter03.Sharpen(imgin)
        elif option == "Gradient":
            imgout = Chapter03.Gradient(imgin)
        elif option == "Median Filter":
            imgout = Chapter03.MedianFilter(imgin)
        elif option == "Spectrum":
            imgout = Chapter04.Spectrum(imgin)
        elif option == "Frequency Filter":
            imgout = Chapter04.FrequencyFilter(imgin)
        elif option == "Remove Moire":
            imgout = Chapter04.RemoveMoire(imgin)
        elif option == "Create Motion Noise":
            imgout = Chapter05.CreateMotionNoise(imgin)
        elif option == "Denoise Motion":
            imgout = Chapter05.DenoiseMotion(imgin)
        elif option == "Connected Component":
            imgout = Chapter09.ConnectedComponent(imgin)
        elif option == "Count Rice":
            imgout = Chapter09.CountRice(imgin)
        else:
            st.write("")

    with col2:
        if option == "None":
            image = Image.open(image_path)
            img_array = np.array(image)
            cv2.imwrite('image_processing/output/out.jpg', cv2.cvtColor(img_array, cv2.IMREAD_GRAYSCALE))
            st.image(image, caption='Output', use_column_width=True)
        else:
            cv2.imwrite('image_processing/output/out.jpg', cv2.cvtColor(imgout, cv2.IMREAD_GRAYSCALE))
            st.image(imgout, caption="Final", use_column_width=True)
    st.write("Link study more: ", "https://samirkhanal35.medium.com/negative-image-6cb7d5edcb54")