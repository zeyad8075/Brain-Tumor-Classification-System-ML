import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# استخدام التخزين المؤقت لمنع إعادة تحميل النموذج مع كل تفاعل للمستخدم
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_trained_model.h5")

try:
    model = load_model()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
    st.stop()

def predict_image(image):
    # تحويل الصورة إلى RGB في حال كانت بصيغة أخرى لتجنب أخطاء الأبعاد
    image = image.convert('RGB').resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return labels[class_idx], prediction[0]

# واجهة المستخدم
st.set_page_config(page_title="تطبيق تحليل الصور الطبية", page_icon="📷")
st.title("📷 تطبيق تحليل الصور الطبية")
st.markdown("قم برفع صورة طبية ليقوم نموذج الذكاء الاصطناعي بتحليلها.")

uploaded_file = st.file_uploader("📂 اختر صورة للتحليل", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("🔍 تحليل الصورة", type="primary"):
        with st.spinner('جاري التحليل...'):
            result, probabilities = predict_image(image)
            
            st.success(f"📌 النتيجة: **{result}**")
            
            st.subheader("📊 الاحتمالات:")
            labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
            for label, prob in zip(labels, probabilities):
                st.write(f"- **{label}**: {prob * 100:.2f}%")
