from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

CLEANED_DATA_PATH = Path("data/processed/mental_health_cleaned.csv")
FIGURE_DIR = Path("outputs/figures")

FEATURE_COLUMNS = [
    "Age",
    "Daily_Screen_Time_hrs",
    "Sleep_Quality_1_10",
    "Stress_Level_1_10",
    "Days_Without_Social_Media",
    "Exercise_Frequency_week",
]

TARGET_COLUMN = "Happiness_Index_3highest"
CLASS_LABELS = ["Low", "Medium", "High"]


def prepare_train_test_data(df):
    """
    Select model features and split the dataset into
    stratified training and testing sets.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )


def save_label_distribution_plot(labels, title, filename):
    """Save a bar chart showing the distribution of happiness labels."""
    label_order = [1, 2, 3]
    counts = labels.value_counts().reindex(label_order)

    plt.figure()
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Happiness level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename)
    plt.close()


def save_screen_time_distribution(X_train):
    """Save a histogram showing daily screen time distribution in training data."""
    plt.figure()
    plt.hist(X_train["Daily_Screen_Time_hrs"], bins=20)
    plt.title("Distribution of Daily Screen Time")
    plt.xlabel("Daily screen time (hours)")
    plt.ylabel("Number of people")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "screen_time_distribution.png")
    plt.close()


def save_stress_by_happiness_plot(X_train, y_train):
    """Save a boxplot comparing stress level across happiness categories."""
    plt.figure()
    sns.boxplot(x=y_train, y=X_train["Stress_Level_1_10"])
    plt.title("Stress Level by Happiness Category")
    plt.xlabel("Happiness level")
    plt.ylabel("Stress level")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "stress_by_happiness.png")
    plt.close()


def train_decision_tree(X_train, y_train):
    """Train a decision tree classifier to predict happiness category."""
    model = DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


def save_confusion_matrix(model, X_test, y_test):
    """Save a confusion matrix for the trained decision tree."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_LABELS,
    )

    disp.plot()
    plt.title("Confusion Matrix - Decision Tree")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png")
    plt.close()

    return y_pred


def save_decision_tree_plot(model):
    """Save a visualization of the trained decision tree."""
    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=FEATURE_COLUMNS,
        class_names=CLASS_LABELS,
        filled=True,
    )
    plt.title("Decision Tree for Predicting Happiness")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "decision_tree.png")
    plt.close()


def main():
    """Train and evaluate a decision tree classifier on the mental health dataset."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEANED_DATA_PATH)

    X_train, X_test, y_train, y_test = prepare_train_test_data(df)

    save_label_distribution_plot(
        y_train,
        "Training Label Distribution",
        "training_label_distribution.png",
    )

    save_label_distribution_plot(
        y_test,
        "Testing Label Distribution",
        "testing_label_distribution.png",
    )

    save_screen_time_distribution(X_train)
    save_stress_by_happiness_plot(X_train, y_train)

    model = train_decision_tree(X_train, y_train)
    y_pred = save_confusion_matrix(model, X_test, y_test)
    save_decision_tree_plot(model)

    print("Decision tree model complete.")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

    print(f"\nFigures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
