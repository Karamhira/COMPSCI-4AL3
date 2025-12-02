import pandas as pd


df = pd.read_csv("./dataset/MN-DS-news-classification.csv")

print("=== Level 1 Category Counts ===")
lvl1_counts = df['category_level_1'].value_counts()
print(lvl1_counts)

print("\n=== Level 2 Counts per Level 1 Category ===")
lvl2_counts = df.groupby('category_level_1')['category_level_2'].value_counts()
print(lvl2_counts)

# Optional: show Level 2 categories with fewer than 50 samples
print("\n=== Level 2 Categories with <50 Samples ===")
small_lvl2 = lvl2_counts[lvl2_counts < 50]
print(small_lvl2)
