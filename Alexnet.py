import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Métrica a ser monitorada
    patience=10,             # Número de épocas sem melhora para parar
    restore_best_weights=True, # Restaura os pesos do modelo da melhor época
    verbose=1
)

def gerar_grafico_ascii(series, titulo, largura_max=60):
    min_val = min(series)
    max_val = max(series)
    intervalo = max_val - min_val
    if intervalo == 0:
        intervalo = 1.0
    grafico_str = f"--- {titulo} ---\n"
    for i, valor in enumerate(series):
        largura_barra = int(((valor - min_val) / intervalo) * (largura_max - 1))
        barra = '#' * largura_barra
        linha = f"Época {i+1:02d} [{barra:<{largura_max}}] {valor:.4f}\n"
        grafico_str += linha
    grafico_str += "\n"
    return grafico_str

# HIPERPARAMETROS
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 50

script_dir = os.path.dirname(os.path.abspath(__file__))
base_output_dir = os.path.join(script_dir, "experimentos")
os.makedirs(base_output_dir, exist_ok=True)
try:
    existing_runs = [int(d) for d in os.listdir(base_output_dir) if d.isdigit()]
    next_run_number = max(existing_runs) + 1 if existing_runs else 1
except Exception as e:
    print(f"Erro ao verificar pastas existentes, começando com a execução 1. Erro: {e}")
    next_run_number = 1
output_dir = os.path.join(base_output_dir, str(next_run_number))
os.makedirs(output_dir)
print(f"Iniciando nova execução. Os resultados serão salvos em: {output_dir}")

# DATASET
base_dir = "C:/Users/Lissa G. Kawasaki/Documents/UNESPAR/3o_Ano/Inteligencia_Artificial/Trabalho3Bim/Dataset"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
IMG_SIZE = (227, 227)
train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(validation_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
class_names = train_ds.class_names
print("Classes:", class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# MODELO
data_augmentation = tf.keras.Sequential([ tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1), tf.keras.layers.RandomZoom(0.1),])
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(227, 227, 3)), data_augmentation, tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu'), tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'), tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)), tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'), tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'), tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Flatten(), tf.keras.layers.Dense(4096, activation='relu'), tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(4096, activation='relu'), tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')])

# LEARNIGN RATE
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=1e-6, verbose=1)


# TREINAMENTO
history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[lr_scheduler, early_stopping])

y_true, y_pred_probs = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred_probs.extend(preds)

# CALCULOS DE MÉTRICAS
y_pred = np.argmax(y_pred_probs, axis=1)

class_report = classification_report(y_true, y_pred, target_names=class_names)
cm = confusion_matrix(y_true, y_pred)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
grafico_acuracia_treino = gerar_grafico_ascii(acc, "Acurácia de Treino")
grafico_acuracia_val = gerar_grafico_ascii(val_acc, "Acurácia de Validação")
grafico_perda_treino = gerar_grafico_ascii(loss, "Perda de Treino")
grafico_perda_val = gerar_grafico_ascii(val_loss, "Perda de Validação")
path_relatorio = os.path.join(output_dir, "relatorio_completo.txt")

# (MODIFICADO) Bloco de escrita do relatório
with open(path_relatorio, "w", encoding="utf-8") as f:
    f.write("="*70 + "\n")
    f.write(f"RELATÓRIO DE TREINAMENTO DO MODELO\n")
    f.write(f"Execução Número: {next_run_number}\n")
    f.write("="*70 + "\n\n")

    f.write("--- HIPERPARÂMETROS DO TREINAMENTO ---\n")
    f.write(f"i.   Taxa de Aprendizagem Inicial: {LEARNING_RATE}\n")
    f.write(f"ii.  Tamanho do Batch (Batch Size): {BATCH_SIZE}\n")
    f.write(f"iii. Número de Épocas Programadas: {EPOCHS}\n\n")

    f.write("--- RESUMO DO DESEMPENHO FINAL ---\n")
    f.write(f"Acurácia final de Treino:    {acc[-1]:.4f}\n")
    f.write(f"Acurácia final de Validação: {val_acc[-1]:.4f}\n")
    f.write(f"Perda final de Treino:       {loss[-1]:.4f}\n")
    f.write(f"Perda final de Validação:    {val_loss[-1]:.4f}\n\n")
    
    f.write("--- MATRIZ DE CONFUSÃO ---\n")
    header = "Previsto -> " + " ".join([f"{name:<8}" for name in class_names])
    f.write(header + "\n")
    for i, name in enumerate(class_names):
        row_str = f"Verdadeiro {name:<8}"
        for val in cm[i]:
            row_str += f"{val:<8}"
        f.write(row_str + "\n")
    f.write("\n")
    
    f.write("--- RELATÓRIO DE CLASSIFICAÇÃO ---\n")
    f.write(class_report)
    f.write("\n")
    
    f.write("--- GRÁFICOS DE TREINAMENTO (TEXTO) ---\n\n")
    f.write(grafico_acuracia_treino)
    f.write(grafico_acuracia_val)
    f.write(grafico_perda_treino)
    f.write(grafico_perda_val)
    
    f.write("="*70 + "\n")
    f.write("FIM DO RELATÓRIO\n")
    f.write("="*70 + "\n")

print(f"\nProcesso concluído! '{output_dir}'.")