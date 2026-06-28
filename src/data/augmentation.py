"""Data augmentation utilities for generating training data."""

import pandas as pd
import random
from pathlib import Path

random.seed(42)

TEMPLATES = {
    'definition': [
        "What is {concept}?",
        "Define {concept}.",
        "What does {concept} mean?",
        "Explain the concept of {concept}.",
        "How is {concept} defined?",
        "What are the main features of {concept}?",
    ],
    'comparison': [
        "How do {a} and {b} differ?",
        "What is the difference between {a} and {b}?",
        "Compare {a} and {b}.",
        "What are the similarities between {a} and {b}?",
        "How does {a} differ from {b}?",
    ],
    'example': [
        "Give an example of {concept}.",
        "What is an instance of {concept}?",
        "How does {concept} manifest itself?",
        "Can you provide a concrete example of {concept}?",
        "Where can we see {concept} in practice?",
    ],
    'deepening': [
        "What are the implications of {concept}?",
        "Analyze {concept} in depth.",
        "What are the strengths and weaknesses of {concept}?",
        "How has {concept} evolved over time?",
        "What are the main critiques of {concept}?",
    ],
    'deep_dive': [
        "What are the implications of {concept}?",
        "Analyze {concept} in depth.",
        "What are the strengths and weaknesses of {concept}?",
        "How has {concept} evolved over time?",
        "What are the main critiques of {concept}?",
    ],
    'quiz': [
        "Quiz me on {concept}.",
        "Test me on {concept}.",
        "Ask me something about {concept}.",
        "I want to be tested on {concept}.",
        "Create a test on {concept}.",
        "Give me a question about {concept}.",
    ]
}

PHILOSOPHERS = [
    "Plato", "Aristotle", "Socrates", "Descartes", "Kant",
    "Nietzsche", "Hegel", "Marx", "Hume", "Locke",
    "Rousseau", "Spinoza", "Leibniz", "Schopenhauer", "Heidegger",
    "Wittgenstein", "Husserl", "Sartre", "Foucault", "Arendt"
]

CONCEPTS = [
    "the Forms", "the categorical imperative", "the social contract",
    "the will to power", "the eternal return", "the cogito",
    "the veil of ignorance", "the state of nature", "the Übermensch",
    "the dialectic", "the self", "the Other", "biopolitics",
    "the phenomenology of spirit", "the invisible hand",
    "the general will", "the principle of utility", "theoria",
    "mimesis", "catharsis", "the unconscious", "the death of God",
    "the end of history", "the iron cage", "the panopticon"
]


def augment_dataset(input_path, output_path, train_path, test_path, test_size=0.2):
    df = pd.read_csv(input_path)
    augmented = []
    for _, row in df.iterrows():
        augmented.append(row.to_dict())
        label = row['label']
        if label == 'comparison':
            parts = row['question'].replace('?', '').split(' e ')
            if len(parts) >= 2:
                for template in TEMPLATES[label][:3]:
                    augmented.append({'question': template.format(a=parts[0], b=parts[1]), 'label': label})
        else:
            for concept in CONCEPTS[:3]:
                for template in TEMPLATES[label][:2]:
                    augmented.append({'question': template.format(concept=concept), 'label': label})
    for philosopher in PHILOSOPHERS[:10]:
        for template in TEMPLATES['quiz'][:3]:
            augmented.append({'question': template.format(concept=philosopher), 'label': 'quiz'})

    aug_df = pd.DataFrame(augmented).drop_duplicates(subset=['question'])
    aug_df = aug_df.sample(frac=1, random_state=42).reset_index(drop=True)
    aug_df.to_csv(output_path, index=False)

    train_df = aug_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(frac=(1 - test_size), random_state=42)
    ).reset_index(drop=True)
    test_df = aug_df[~aug_df.index.isin(train_df.index)].reset_index(drop=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Original: {len(df)} | Augmented: {len(aug_df)}")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    for label in sorted(aug_df['label'].unique()):
        print(f"  {label}: train={len(train_df[train_df['label']==label])}, test={len(test_df[test_df['label']==label])}")


def augment_dataframe(df: pd.DataFrame, n_concepts: int = 3,
                      n_philosophers: int = 10) -> pd.DataFrame:
    """Generate augmented questions from a DataFrame without including originals.
    
    Use this to augment only the training set after the train/test split,
    eliminating data leakage.
    """
    augmented = []
    for _, row in df.iterrows():
        label = row['label']
        if label == 'comparison':
            parts = row['question'].replace('?', '').split(' e ')
            if len(parts) >= 2:
                for template in TEMPLATES[label][:3]:
                    augmented.append({'question': template.format(a=parts[0], b=parts[1]), 'label': label})
        else:
            for concept in CONCEPTS[:n_concepts]:
                for template in TEMPLATES[label][:2]:
                    augmented.append({'question': template.format(concept=concept), 'label': label})
    for philosopher in PHILOSOPHERS[:n_philosophers]:
        for template in TEMPLATES['quiz'][:3]:
            augmented.append({'question': template.format(concept=philosopher), 'label': 'quiz'})
    aug_df = pd.DataFrame(augmented).drop_duplicates(subset=['question'])
    aug_df = aug_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return aug_df


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent.parent
    augment_dataset(
        str(base / 'data' / 'labels' / 'questions_labeled.csv'),
        str(base / 'data' / 'labels' / 'questions_augmented.csv'),
        str(base / 'data' / 'labels' / 'questions_train.csv'),
        str(base / 'data' / 'labels' / 'questions_test.csv')
    )
