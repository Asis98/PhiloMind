"""
Data augmentation per il dataset di classificazione delle domande.
Genera variazioni semantiche delle domande etichettate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import csv

# Mapping di sinonimi e variazioni
VARIATION_TEMPLATES = {
    'definizione': [
        "Cos'è {}?",
        "Che cosa si intende per '{}'?",
        "Qual è la definizione di {}?",
        "Spiega il concetto di {}.",
        "Definisci {}.",
        "Come definiresti {}?",
    ],
    'confronto': [
        "Come si differenziano {} e {}?",
        "Qual è la differenza tra {} e {}?",
        "Metti a confronto {} e {}.",
        "In cosa si contrappongono {} e {}?",
        "Quali sono le similarità e differenze tra {} e {}?",
        "Come si distinguono {} da {}?",
    ],
    'esempio': [
        "Puoi fare un esempio di {}?",
        "Come si manifesta {} nella vita reale?",
        "Dammi un esempio concreto di {}.",
        "Come si applica {}?",
        "Illustra {} con un esempio.",
        "Fai un caso pratico di {}.",
    ],
    'approfondimento': [
        "Approfondisci il tema di {}.",
        "Sviluppa il concetto di {}.",
        "Analizza in dettaglio {}.",
        "Quali sono gli aspetti principali di {}?",
        "Come si evolve il pensiero su {}?",
        "Esamina criticamente {}.",
    ],
    'quiz': [
        "Mettimi alla prova su {}.",
        "Interrogami su {}.",
        "Voglio essere testato su {}.",
        "Fai una domanda di verifica su {}.",
        "Crea un quiz su {}.",
        "Chiedi qualcosa su {}.",
    ]
}

# Sinonimi e variazioni di concept chiave
CONCEPT_VARIATIONS = {
    # Filosofi
    'Platone': ['Plato', 'il filosofo Platone'],
    'Aristotele': ['Aristotele', 'il Filosofo'],
    'Descartes': ['Cartesio', 'Descartes'],
    'Nietzsche': ['Friedrich Nietzsche'],
    'Marx': ['Karl Marx'],
    'Hegel': ['Georg Wilhelm Friedrich Hegel'],
    'Kant': ['Immanuel Kant'],
    'Rousseau': ['Jean-Jacques Rousseau'],
    'Hobbes': ['Thomas Hobbes'],
    'Spinoza': ['Baruch Spinoza'],
    'Locke': ['John Locke'],
    'Hume': ['David Hume'],
    'Sartre': ['Jean-Paul Sartre'],
    'Wittgenstein': ['Ludwig Wittgenstein'],

    # Concetti
    'il contratto sociale': ['contratto sociale di', 'patto sociale'],
    'volontà di potenza': ['voluntas potentiae', 'volontà di potenza'],
    'il mito della caverna': ['mito della caverna', 'allegoria della caverna'],
    'imperativo categorico': ['imperativo categorico', 'deber assoluto'],
}

def load_original_data(csv_path):
    """Carica il CSV originale."""
    return pd.read_csv(csv_path)

def extract_key_concepts(question):
    """Estrae concetti chiave da una domanda."""
    concepts = []
    for concept, variations in CONCEPT_VARIATIONS.items():
        if concept.lower() in question.lower() or any(v.lower() in question.lower() for v in variations):
            concepts.append(concept)
    return concepts if concepts else ['filosofia']  # fallback

def generate_augmented_questions(df, multiplier=2):
    """Genera domande augmentate usando template variations."""
    augmented_rows = []

    for idx, row in df.iterrows():
        original_question = row['question']
        label = row['label']

        # Mantieni l'originale
        augmented_rows.append(row.to_dict())

        # Genera variazioni solo per i dati originali non troppo lunghi
        if len(original_question) < 150:
            concepts = extract_key_concepts(original_question)

            # Prendi 1-2 template a caso per questa classe
            templates = VARIATION_TEMPLATES.get(label, [])
            if templates:
                # Scegli 1-2 template
                n_variations = min(2, len(templates))
                selected_templates = np.random.choice(templates, n_variations, replace=False)

                for template in selected_templates:
                    # Sostituisci i placeholder nei template
                    try:
                        if '{}' in template:
                            # Per confronto abbiamo 2 {}
                            if label == 'confronto' and len(concepts) >= 2:
                                concept1, concept2 = concepts[0], concepts[1]
                                new_question = template.format(concept1, concept2)
                            elif len(concepts) > 0:
                                new_question = template.format(concepts[0])
                            else:
                                new_question = template.format('questo concetto')

                            augmented_rows.append({
                                'question': new_question,
                                'label': label
                            })
                    except Exception as e:
                        print(f"Errore generando variazione: {e}")
                        pass

    return pd.DataFrame(augmented_rows)

def validate_dataset(df):
    """Valida la qualità del dataset."""
    print("\n📊 STATISTICHE DATASET:")
    print(f"Total rows: {len(df)}")
    print(f"\nDistribuzione classi:")
    print(df['label'].value_counts())
    print(f"\nLunghezza media domande per classe:")
    print(df.groupby('label')['question'].apply(lambda x: x.str.len().mean()).round(2))

    # Check duplicati
    duplicates = df['question'].duplicated().sum()
    print(f"\nDuplicate questions: {duplicates}")

    return df

def save_splits(df, output_dir, train_ratio=0.8):
    """Salva train/test split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split stratificato per mantenere proporzioni
    train_size = int(len(df) * train_ratio)

    # Stratificazione per classe
    train_indices = []
    test_indices = []

    for label in df['label'].unique():
        label_indices = df[df['label'] == label].index.tolist()
        n_train = int(len(label_indices) * train_ratio)

        np.random.seed(42)
        np.random.shuffle(label_indices)

        train_indices.extend(label_indices[:n_train])
        test_indices.extend(label_indices[n_train:])

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # Save
    train_df.to_csv(output_dir / 'questions_train.csv', index=False)
    test_df.to_csv(output_dir / 'questions_test.csv', index=False)
    df.to_csv(output_dir / 'questions_augmented.csv', index=False)

    print(f"\n✅ DATASET SALVATI:")
    print(f"Train set: {len(train_df)} domande → {output_dir / 'questions_train.csv'}")
    print(f"Test set: {len(test_df)} domande → {output_dir / 'questions_test.csv'}")
    print(f"Full augmented: {len(df)} domande → {output_dir / 'questions_augmented.csv'}")

    return train_df, test_df

if __name__ == '__main__':
    # Path
    csv_path = 'data/labels/questions_labeled.csv'
    output_dir = 'data/labels'

    # Load original
    print("📖 Caricamento dataset originale...")
    df_original = load_original_data(csv_path)
    print(f"Loaded {len(df_original)} questions")

    # Augment
    print("\n🔄 Augmentazione dati...")
    df_augmented = generate_augmented_questions(df_original, multiplier=2)

    # Rimuovi duplicati esatti
    df_augmented = df_augmented.drop_duplicates(subset=['question']).reset_index(drop=True)

    # Validate
    validate_dataset(df_augmented)

    # Split
    train_df, test_df = save_splits(df_augmented, output_dir, train_ratio=0.8)

