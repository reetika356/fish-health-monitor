import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------- Config ----------------
MODEL_PATH = "fish_model_v2.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Healthy", "Sick"]  # index 0 = Healthy, index 1 = Sick (alphabetical, matches training)

st.set_page_config(page_title="Fish Health Monitor", page_icon="🐟")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def send_email_alert(sender_email, sender_password, receiver_email, confidence, filename):
    subject = "Fish Health Alert: Sick Fish Detected"
    body = (
        f"A sick fish has been detected in the uploaded image: {filename}\n"
        f"Model confidence: {confidence:.1%}\n\n"
        f"Please check the tank as soon as possible."
    )

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)


# ---------------- UI ----------------
st.title("🐟 Fish Health Monitoring System")
st.write("Upload a photo of a fish and get an instant health prediction.")

with st.sidebar:
    st.header("Email Alert Settings (optional)")
    enable_email = st.checkbox("Send email alert if fish is Sick")
    sender_email = st.text_input("Sender Gmail address") if enable_email else None
    sender_password = st.text_input(
        "Sender Gmail App Password", type="password",
        help="Use a 16-character Gmail App Password, not your normal password. "
             "Generate one at myaccount.google.com/apppasswords"
    ) if enable_email else None
    receiver_email = st.text_input("Alert recipient email") if enable_email else None

uploaded_file = st.file_uploader("Upload fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    model = load_model()
    x = preprocess_image(img)
    prob_sick = float(model.predict(x, verbose=0)[0][0])

    predicted_idx = 1 if prob_sick >= 0.5 else 0
    label = CLASS_NAMES[predicted_idx]
    confidence = prob_sick if predicted_idx == 1 else (1 - prob_sick)

    if label == "Sick":
        st.error(f"⚠️ Prediction: **Sick** (confidence: {confidence:.1%})")
    else:
        st.success(f"✅ Prediction: **Healthy** (confidence: {confidence:.1%})")

    st.caption(
        "Note: this model was trained on a small (305-image) dataset and is a prototype, "
        "not a clinically validated diagnostic tool."
    )

    if label == "Sick" and enable_email:
        if sender_email and sender_password and receiver_email:
            try:
                send_email_alert(sender_email, sender_password, receiver_email, confidence, uploaded_file.name)
                st.info(f"📧 Email alert sent to {receiver_email}")
            except Exception as e:
                st.warning(f"Email alert failed to send: {e}")
        else:
            st.warning("Fill in all email fields in the sidebar to enable alerts.")
