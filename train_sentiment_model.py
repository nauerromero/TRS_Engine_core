"""
Training Script - Sentiment Model
Entrena modelo BERT para clasificación de emociones en 3 clases
Optimizado para GPU local (Alienware M18 R2)
"""

import json
import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os

print("=" * 60)
print("ENTRENAMIENTO DE MODELO - SENTIMENT ANALYSIS")
print("=" * 60)

# Verificar GPU
print("\n[SETUP] Verificando GPU...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"   GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("   WARNING: GPU no disponible, usando CPU (sera lento)")

# Cargar datos
print("\n[STEP 1/6] Cargando dataset...")
with open('Data/sentiment_training/labeled_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels_str = [item['label'] for item in data]

# Mapear labels a números
label2id = {'positive': 0, 'neutral': 1, 'negative': 2}
id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}
labels = [label2id[label] for label in labels_str]

print(f"   Total ejemplos: {len(texts)}")
print(f"   Clases: {list(label2id.keys())}")

# Split dataset
print("\n[STEP 2/6] Dividiendo dataset (70% train, 15% val, 15% test)...")
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"   Train: {len(X_train)} ejemplos")
print(f"   Validation: {len(X_val)} ejemplos")  
print(f"   Test: {len(X_test)} ejemplos")

# Cargar modelo y tokenizer
print("\n[STEP 3/6] Cargando BERT base...")
try:
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    print(f"   Modelo cargado: {model_name}")
    print(f"   Parametros: ~110M")
    print(f"   Device: {device}")
    
except Exception as e:
    print(f"   ERROR: {e}")
    print("\n   Instala transformers:")
    print("   pip install transformers torch --index-url https://download.pytorch.org/whl/cu121")
    exit(1)

# Tokenizar
print("\n[STEP 4/6] Tokenizando textos...")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# Crear Dataset class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, y_train)
val_dataset = SentimentDataset(val_encodings, y_val)
test_dataset = SentimentDataset(test_encodings, y_test)

print(f"   Datasets creados OK")

# Configurar entrenamiento
print("\n[STEP 5/6] Configurando entrenamiento...")
training_args = TrainingArguments(
    output_dir='./Models/bert-sentiment-training',
    num_train_epochs=5,  # Más epochs por dataset pequeño
    per_device_train_batch_size=16,  # Ajustar según VRAM
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),  # Precision mixta si hay GPU
    report_to="none",  # No usar wandb por ahora
)

# Función de métricas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

# Crear Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   FP16 (precision mixta): {training_args.fp16}")

# ENTRENAR!
print("\n[STEP 6/6] INICIANDO ENTRENAMIENTO...")
print("   (Esto puede tomar 30-60 minutos con GPU)")
print("   Puedes dejar esto corriendo y dormir!")
print("=" * 60)

start_time = datetime.now()

try:
    trainer.train()
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ENTRENAMIENTO COMPLETADO!")
    print(f"Duracion: {training_duration:.1f} minutos")
    print("=" * 60)
    
    # Evaluar en test set
    print("\n[EVALUATION] Evaluando en test set...")
    test_results = trainer.predict(test_dataset)
    test_preds = test_results.predictions.argmax(-1)
    
    # Métricas
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average='macro')
    
    print(f"\n[METRICS] Resultados finales:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1-Score (macro): {test_f1:.4f}")
    
    # Classification report
    print(f"\n[REPORT] Por clase:")
    print(classification_report(y_test, test_preds, target_names=['positive', 'neutral', 'negative']))
    
    # Confusion matrix
    print(f"[CONFUSION MATRIX]")
    cm = confusion_matrix(y_test, test_preds)
    print("              Predicted")
    print("              pos  neu  neg")
    for i, label in enumerate(['positive ', 'neutral  ', 'negative ']):
        print(f"Actual {label}: {cm[i]}")
    
    # Guardar modelo
    print(f"\n[SAVING] Guardando modelo...")
    model_path = "Models/bert-sentiment-trs"
    os.makedirs(model_path, exist_ok=True)
    
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Guardar métricas
    metrics = {
        "model": "bert-base-uncased",
        "num_labels": 3,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "training_duration_minutes": training_duration,
        "training_date": datetime.now().isoformat(),
        "device": str(device),
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
    }
    
    with open(f"{model_path}/training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   Modelo guardado en: {model_path}/")
    print(f"   Metricas guardadas en: {model_path}/training_metrics.json")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("[SUMMARY] MODELO LISTO PARA USO")
    print("=" * 60)
    print(f"Accuracy: {test_acc*100:.1f}%")
    print(f"F1-Score: {test_f1:.3f}")
    print(f"Modelo: {model_path}/")
    print("\nProximos pasos:")
    print("1. Revisar metricas en training_metrics.json")
    print("2. Integrar modelo en sentiment_model.py")
    print("3. Probar en pipeline completo")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] Error durante entrenamiento: {e}")
    import traceback
    traceback.print_exc()
    print("\nRevisa:")
    print("1. GPU disponible (nvidia-smi)")
    print("2. VRAM suficiente (reduce batch_size si falla)")
    print("3. Dependencias instaladas correctamente")

