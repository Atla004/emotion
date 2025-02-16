import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from keras.api.models import Sequential, load_model
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.optimizers import Adam
from keras.api.utils import to_categorical
from keras.api.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator


import pandas as pd
import os


# Configuración inicial
EMOCIONES = ["Enojo", "Asco", "Miedo", "Feliz", "Triste", "Neutral", "Sorpresa"]
MODELO_ARCHIVO = "modelo_emociones.h5"
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

class AplicacionEmociones:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Detección de Emociones Faciales")
        
        # Configurar modelo
        if not os.path.exists(MODELO_ARCHIVO):
            self.entrenar_modelo()
            
        self.modelo = load_model(MODELO_ARCHIVO)
        self.cascada_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.captura = None
        self.en_ejecucion = False

        # Interfaz gráfica
        self.configurar_interfaz()
        
    def configurar_interfaz(self):
        self.canvas = tk.Canvas(self.ventana, width=640, height=480)
        self.canvas.pack()
        
        self.etiqueta_emocion = ttk.Label(self.ventana, text="Emoción detectada: ", font=('Helvetica', 14))
        self.etiqueta_emocion.pack(pady=10)
        
        self.boton_inicio = ttk.Button(self.ventana, text="Iniciar Cámara", command=self.iniciar_camara)
        self.boton_inicio.pack(side=tk.LEFT, padx=10)
        
        self.boton_detener = ttk.Button(self.ventana, text="Detener Cámara", command=self.detener_camara)
        self.boton_detener.pack(side=tk.RIGHT, padx=10)
    
    def entrenar_modelo(self):
        print("Entrenando nuevo modelo...")
        
        # Configurar generadores de datos
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=64,
            class_mode='categorical',
            shuffle=True
        )

        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )

        # Crear modelo
        modelo = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        modelo.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Callback para early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        historia = modelo.fit(
            train_generator,
            epochs=50,
            validation_data=test_generator,
            callbacks=[early_stop]
        )
        
        modelo.save(MODELO_ARCHIVO)
    
    # Métodos restantes iguales
    def iniciar_camara(self):
        self.captura = cv2.VideoCapture(0)
        self.en_ejecucion = True
        self.actualizar_frame()
    
    def detener_camara(self):
        self.en_ejecucion = False
        if self.captura:
            self.captura.release()
    
    def preprocesar_imagen(self, cara):
        cara = cv2.resize(cara, (48,48))
        cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
        cara = np.expand_dims(cara, axis=-1)
        cara = np.expand_dims(cara, axis=0)
        return cara / 255.0
    
    def actualizar_frame(self):
        if self.en_ejecucion:
            ret, frame = self.captura.read()
            if ret:
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                caras = self.cascada_caras.detectMultiScale(gris, 1.3, 5)
                
                for (x,y,w,h) in caras:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cara = frame[y:y+h, x:x+w]
                    cara_procesada = self.preprocesar_imagen(cara)
                    prediccion = self.modelo.predict(cara_procesada)
                    emocion = EMOCIONES[np.argmax(prediccion)]
                    self.etiqueta_emocion.config(text=f"Emoción detectada: {emocion}")
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk
                self.canvas.create_image(0,0, anchor=tk.NW, image=imgtk)
            
            self.ventana.after(10, self.actualizar_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionEmociones(root)
    root.mainloop()