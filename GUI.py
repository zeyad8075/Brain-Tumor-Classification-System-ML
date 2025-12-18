import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)
from PyQt6.QtGui import QPixmap, QFont, QIcon, QColor
from PyQt6.QtCore import Qt, QSize
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "mobilenet_trained_model.h5"
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("تم تحميل النموذج بنجاح!")
except Exception as e:
    print(f"حدث خطأ أثناء تحميل النموذج: {e}")
    sys.exit()


def predict_image(image_path):
    try:
        image = Image.open(image_path).resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0) 
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        return labels[class_idx], prediction[0]
    except Exception as e:
        print(f"حدث خطأ أثناء التحليل: {e}")
        return "حدث خطأ أثناء التحليل", None


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("📷 تطبيق تحليل الصور الطبية")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #E8F5E9;") 

        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(14)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(20)

        title_label = QLabel("تطبيق تحليل الصور الطبية")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2E7D32;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        self.button = QPushButton("📂 اختر صورة للتحليل")
        self.button.setFont(font)
        self.button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        self.button.setIconSize(QSize(24, 24))
        self.button.clicked.connect(self.upload_image)
        main_layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel("🔍 سيتم عرض الصورة هنا")
        self.image_label.setFont(font)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: white;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 10px;
            }
        """
        )
        main_layout.addWidget(self.image_label)

        result_frame = QWidget()
        result_frame.setStyleSheet(
            """
            QWidget {
                background-color: white;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 15px;
            }
        """
        )
        result_layout = QVBoxLayout(result_frame)
        result_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_label = QLabel("📌 النتيجة: لم يتم التحليل بعد")
        self.result_label.setFont(font)
        self.result_label.setStyleSheet("color: #2E7D32;")
        result_layout.addWidget(self.result_label)

        self.probabilities_label = QLabel("📊 الاحتمالات: غير متاحة")
        self.probabilities_label.setFont(font)
        self.probabilities_label.setStyleSheet("color: #555;")
        result_layout.addWidget(self.probabilities_label)

        main_layout.addWidget(result_frame)

        self.setLayout(main_layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "اختر صورة", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            print(f"تم اختيار الصورة: {file_path}")

            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

            result, probabilities = predict_image(file_path)
            if result == "حدث خطأ أثناء التحليل":
                QMessageBox.critical(self, "خطأ", "حدث خطأ أثناء تحليل الصورة. يرجى المحاولة مرة أخرى.")
            else:

                self.result_label.setText(f"📌 النتيجة: {result}")
                if probabilities is not None:
                    prob_text = "📊 الاحتمالات:\n"
                    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
                    for label, prob in zip(labels, probabilities):
                        prob_text += f"{label}: {prob * 100:.2f}%\n"
                    self.probabilities_label.setText(prob_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())