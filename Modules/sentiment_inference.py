"""
Sentiment Inference Module
Usa el modelo BERT entrenado para predecir emociones en respuestas de candidatos
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

class SentimentPredictor:
    """
    Predictor de sentimiento usando modelo BERT entrenado
    """
    
    def __init__(self, model_path="Models/bert-sentiment-saori", fallback_model="nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Inicializa el predictor cargando el modelo y tokenizer
        
        Args:
            model_path: Ruta al modelo entrenado local (si existe)
            fallback_model: Modelo público de HuggingFace a usar si el local no existe
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Verificar si el modelo local existe
        model_exists = os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json"))
        
        if model_exists:
            # Usar modelo local si existe
            print(f"[SENTIMENT] Cargando modelo local desde {model_path}...")
            load_path = model_path
        else:
            # Usar modelo público de HuggingFace como fallback
            print(f"[SENTIMENT] Modelo local no encontrado en {model_path}")
            print(f"[SENTIMENT] Usando modelo público de HuggingFace: {fallback_model}")
            load_path = fallback_model
        
        # Cargar modelo y tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.model = BertForSequenceClassification.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()  # Modo evaluación
        
        # Mapeo de labels (compatible con modelos estándar de sentimiento)
        # Si el modelo tiene su propio id2label, usarlo; si no, usar el mapeo por defecto
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            # Normalizar labels a nuestro formato estándar
            label_mapping = {}
            for idx, label in self.id2label.items():
                label_lower = label.lower()
                if 'positive' in label_lower or 'pos' in label_lower or label_lower in ['5', '4']:
                    label_mapping[idx] = 'positive'
                elif 'negative' in label_lower or 'neg' in label_lower or label_lower in ['1', '2']:
                    label_mapping[idx] = 'negative'
                else:
                    label_mapping[idx] = 'neutral'
            self.id2label = label_mapping
        else:
            # Mapeo por defecto para modelos de 3 clases
            self.id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}
        
        self.label2emotion = {
            'positive': ['enthusiastic', 'confident'],
            'neutral': ['neutral'],
            'negative': ['anxious', 'frustrated', 'negative']
        }
        
        print(f"[SENTIMENT] Modelo cargado en {self.device}")
        print(f"[SENTIMENT] Labels disponibles: {list(self.id2label.values())}")
    
    def predict(self, text):
        """
        Predice el sentimiento de un texto
        
        Args:
            text (str): Texto a analizar
            
        Returns:
            dict: {
                'sentiment': 'positive'|'neutral'|'negative',
                'emotion': 'enthusiastic'|'confident'|'neutral'|...,
                'confidence': float (0-1),
                'probabilities': {'positive': float, 'neutral': float, 'negative': float}
            }
        """
        # Tokenizar
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predecir
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Extraer resultados
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
        
        sentiment = self.id2label[predicted_class]
        
        # Convertir probabilidades a dict
        probabilities = {
            self.id2label[i]: probs[0][i].item()
            for i in range(len(self.id2label))
        }
        
        # Seleccionar emoción específica (usa la primera de la lista por ahora)
        emotion = self.label2emotion[sentiment][0]
        
        return {
            'sentiment': sentiment,
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def predict_batch(self, texts):
        """
        Predice el sentimiento de múltiples textos
        
        Args:
            texts (list): Lista de textos
            
        Returns:
            list: Lista de diccionarios con predicciones
        """
        return [self.predict(text) for text in texts]


def demo():
    """
    Función de demostración
    """
    print("=" * 60)
    print("DEMO - Sentiment Predictor")
    print("=" * 60)
    
    # Inicializar predictor
    try:
        predictor = SentimentPredictor()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nEjecuta primero:")
        print("  python train_sentiment_model.py")
        return
    
    # Ejemplos de prueba
    test_texts = [
        "I'm really excited about this opportunity! I love working with these technologies.",
        "I don't know, maybe it's okay. I'm not sure about the requirements.",
        "This is frustrating. I don't think I understand what you're asking.",
        "Yes, I have 5 years of experience with Python and Django.",
        "Honestly, I'm not very confident about this. It seems too complicated."
    ]
    
    print("\n[PREDICTIONS]\n")
    for i, text in enumerate(test_texts, 1):
        result = predictor.predict(text)
        
        print(f"{i}. Text: {text[:60]}...")
        print(f"   Sentiment: {result['sentiment'].upper()}")
        print(f"   Emotion: {result['emotion']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"      {label}: {prob:.2%}")
        print()
    
    print("=" * 60)
    print("[DEMO] Completado. Listo para integración!")
    print("=" * 60)


if __name__ == "__main__":
    demo()

