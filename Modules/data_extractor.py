"""
Data Extractor - TRS Engine Core
Extrae respuestas de entrevistas para entrenar modelo de sentimiento
"""

import pandas as pd
import json
import os
from datetime import datetime

def extract_from_chat_templates():
    """Extraer respuestas de las plantillas del chat_simulator"""
    
    # Templates del sistema actual
    templates = {
        "backend": [
            ("I'd use async/await and proper error handling.", "confident"),
            ("We scaled using load balancers and caching layers.", "confident"),
            ("Yes, I've worked with that in production.", "positive"),
            ("I usually handle that with standard tools.", "neutral"),
            ("I'm not sure, I haven't used that much.", "negative"),
        ],
        "data": [
            ("ETL is essential for transforming raw data into usable formats.", "confident"),
            ("I built pipelines using Airflow and Python.", "positive"),
            ("We used Spark for distributed processing.", "confident"),
            ("I usually handle that with standard tools.", "neutral"),
            ("I'm not sure, I haven't used that much.", "negative"),
        ],
        "admin": [
            ("I use Google Calendar and Trello for scheduling.", "positive"),
            ("I prioritize tasks using urgency and impact.", "confident"),
            ("Yes, I've worked with that in production.", "positive"),
            ("I usually handle that with standard tools.", "neutral"),
            ("I'm not sure, I haven't used that much.", "negative"),
        ]
    }
    
    responses = []
    for domain, texts in templates.items():
        for text, emotion in texts:
            responses.append({
                "text": text,
                "label": emotion,
                "domain": domain,
                "source": "template"
            })
    
    return responses

def extract_from_emotional_states():
    """Generar ejemplos basados en estados emocionales conocidos"""
    
    examples = [
        # Positive/Enthusiastic
        ("I'm really excited about this opportunity!", "positive"),
        ("This sounds absolutely perfect for me!", "positive"),
        ("I'd love to work on this kind of project!", "positive"),
        ("That's exactly the challenge I'm looking for!", "positive"),
        ("I'm confident I can deliver great results here!", "positive"),
        
        # Neutral
        ("I think that could work.", "neutral"),
        ("That seems like a reasonable approach.", "neutral"),
        ("I've heard of that technology before.", "neutral"),
        ("It depends on the specific requirements.", "neutral"),
        ("I would need to evaluate the options.", "neutral"),
        
        # Negative/Anxious
        ("I'm not entirely sure about that.", "negative"),
        ("I haven't had much experience with that.", "negative"),
        ("That sounds quite challenging.", "negative"),
        ("I don't think I'm familiar with that approach.", "negative"),
        ("I'm uncertain about those requirements.", "negative"),
    ]
    
    responses = []
    for text, label in examples:
        responses.append({
            "text": text,
            "label": label,
            "domain": "general",
            "source": "emotional_examples"
        })
    
    return responses

def generate_technical_responses():
    """Generar respuestas técnicas variadas"""
    
    responses = []
    
    # Positivas técnicas
    positive_tech = [
        "I've implemented microservices using Docker and Kubernetes in my last project.",
        "I'm proficient with React hooks and state management patterns.",
        "I have extensive experience with PostgreSQL and query optimization.",
        "I've led teams through complete CI/CD pipeline implementations.",
        "I'm comfortable working with AWS services like Lambda and S3.",
    ]
    
    # Neutrales técnicas
    neutral_tech = [
        "I've used that framework in a couple of projects.",
        "I understand the basic concepts of that technology.",
        "I could learn that tool if needed for the role.",
        "I'm familiar with similar approaches to that problem.",
        "I've read about that pattern but haven't implemented it yet.",
    ]
    
    # Negativas técnicas
    negative_tech = [
        "I haven't worked with that specific technology before.",
        "That's outside my current area of expertise.",
        "I'm still learning about distributed systems.",
        "I find that framework quite complex to work with.",
        "I've struggled with similar implementations in the past.",
    ]
    
    for text in positive_tech:
        responses.append({"text": text, "label": "positive", "domain": "technical", "source": "generated"})
    
    for text in neutral_tech:
        responses.append({"text": text, "label": "neutral", "domain": "technical", "source": "generated"})
    
    for text in negative_tech:
        responses.append({"text": text, "label": "negative", "domain": "technical", "source": "generated"})
    
    return responses

def augment_data(responses, multiplier=2):
    """Data augmentation simple - variaciones de frases"""
    
    augmented = []
    
    variations = {
        "I'm": ["I am", "I'm"],
        "I've": ["I have", "I've"],
        "really": ["very", "really", "quite"],
        "excited": ["excited", "enthusiastic", "eager"],
        "not sure": ["uncertain", "not sure", "unsure"],
        "haven't": ["have not", "haven't"],
        "worked": ["worked", "dealt", "been involved"],
    }
    
    for response in responses:
        # Original
        augmented.append(response)
        
        # Variación 1: cambiar contracciones
        text = response["text"]
        if "I'm" in text:
            aug_text = text.replace("I'm", "I am")
            augmented.append({
                **response,
                "text": aug_text,
                "source": response["source"] + "_aug"
            })
        elif "I've" in text:
            aug_text = text.replace("I've", "I have")
            augmented.append({
                **response,
                "text": aug_text,
                "source": response["source"] + "_aug"
            })
    
    return augmented

def main():
    """Función principal de extracción"""
    
    print("=" * 60)
    print("EXTRACTOR DE DATOS - MODELO DE SENTIMIENTO")
    print("=" * 60)
    
    # Extraer de diferentes fuentes
    print("\n[1/4] Extrayendo respuestas de templates...")
    templates = extract_from_chat_templates()
    print(f"   OK: {len(templates)} respuestas de templates")
    
    print("\n[2/4] Generando ejemplos emocionales...")
    emotional = extract_from_emotional_states()
    print(f"   OK: {len(emotional)} ejemplos emocionales")
    
    print("\n[3/4] Generando respuestas técnicas...")
    technical = generate_technical_responses()
    print(f"   OK: {len(technical)} respuestas tecnicas")
    
    # Combinar todas
    all_responses = templates + emotional + technical
    print(f"\n   Total inicial: {len(all_responses)} ejemplos")
    
    # Augmentation
    print("\n[4/4] Aplicando data augmentation...")
    augmented = augment_data(all_responses, multiplier=2)
    print(f"   OK: {len(augmented)} ejemplos despues de augmentation")
    
    # Normalizar labels a 3 clases
    print("\n[INFO] Normalizando a 3 clases...")
    label_mapping = {
        "enthusiastic": "positive",
        "confident": "positive",
        "positive": "positive",
        "neutral": "neutral",
        "anxious": "negative",
        "frustrated": "negative",
        "negative": "negative"
    }
    
    for response in augmented:
        response["label"] = label_mapping.get(response["label"], "neutral")
    
    # Contar distribución
    from collections import Counter
    label_counts = Counter(r["label"] for r in augmented)
    print(f"\n[DISTRIBUCIÓN]")
    for label, count in sorted(label_counts.items()):
        percentage = count / len(augmented) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Guardar
    os.makedirs("Data/sentiment_training", exist_ok=True)
    output_path = "Data/sentiment_training/labeled_data.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Dataset guardado: {output_path}")
    print(f"   Total: {len(augmented)} ejemplos")
    print(f"   Listo para entrenamiento!")
    
    return augmented

if __name__ == "__main__":
    main()

