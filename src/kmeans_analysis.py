from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

KMEANS_DATA_PATH = Path("data/processed/kmeans_ready.csv")
OUTPUT_DIR = Path("data/processed")


def run_kmeans(df, n_clusters=3):
    """Run K-means clustering and return clustered data and centroids."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(df)

    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    centroids = pd.DataFrame(model.cluster_centers_, columns=df.columns)

    return clustered_df, centroids


def print_cluster_summary(name, clustered_df, centroids):
    """Print cluster centers and cluster sizes."""
    print(f"\n===== {name} =====")
    print("\nCluster centers:")
    print(centroids.round(3))

    print("\nCluster sizes:")
    print(clustered_df["cluster"].value_counts().sort_index())


def main():
    """Run K-means analysis with and without age."""
    df = pd.read_csv(KMEANS_DATA_PATH)

    clustered_with_age, centroids_with_age = run_kmeans(df, n_clusters=3)
    print_cluster_summary("K-means with age", clustered_with_age, centroids_with_age)

    df_without_age = df.drop(columns=["Age"])
    clustered_without_age, centroids_without_age = run_kmeans(
        df_without_age, n_clusters=3
    )
    print_cluster_summary(
        "K-means without age",
        clustered_without_age,
        centroids_without_age,
    )

    clustered_with_age.to_csv(OUTPUT_DIR / "kmeans_clusters_with_age.csv", index=False)
    clustered_without_age.to_csv(
        OUTPUT_DIR / "kmeans_clusters_without_age.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
