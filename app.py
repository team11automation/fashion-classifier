import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

# ===== Model Load Karna =====
model = load_model("fashion_model.keras")

# ===== Categories Ke Naam =====
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ===== Page Setup =====
st.title("👕 Fashion Item Classifier")
st.write("Ek clothing item ki image upload karo, model bata dega kaunsi category hai.")

# ===== Image Upload =====
uploaded_file = st.file_uploader("Please upload your file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.success("Image uploaded successfully")
    st.image(img, caption="Your uploaded image", width=150)

    # ===== Preprocessing =====
    img = img.convert("L")               # Grayscale
    img_array = np.array(img)

    if img_array.mean() > 127:           # Colors invert (agar background safaid hai)
        img_array = 255 - img_array

    img_for_crop = Image.fromarray(img_array)
    bbox = img_for_crop.getbbox()
    if bbox:
        img_for_crop = img_for_crop.crop(bbox)   # Crop (khaali jagah hatao)

    final_img = img_for_crop.resize((28, 28))     # Resize
    final_array = np.array(final_img) / 255.0     # Normalize
    final_array = final_array.reshape(1, 28, 28, 1)  # Reshape

    st.image(final_img.resize((140, 140)), caption="Model ko jo dikh raha hai")

    # ===== Prediction =====
    prediction = model.predict(final_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    # ===== Result =====
    st.success(f"Predicted: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")
    st.bar_chart(prediction[0])
else:
    st.info("Upar image upload karo prediction dekhne ke liye.")