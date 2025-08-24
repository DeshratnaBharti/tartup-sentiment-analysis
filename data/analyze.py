import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/tweets.csv")

# Clean text
df["clean_text"] = df["text"].str.lower()

# Sentiment analysis
df["polarity"] = df["clean_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["sentiment"] = df["polarity"].apply(
    lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")
)

# Brand-wise summary
summary = df.groupby("brand")["sentiment"].value_counts().unstack().fillna(0)

# Save outputs
df.to_csv("processed.csv", index=False)
summary.to_csv("brand_summary.csv")

# Visualization
summary.plot(kind="bar", stacked=True, figsize=(8,5))
plt.title("Startup Sentiment Analysis")
plt.ylabel("Count")
plt.savefig("sentiment_chart.png")

print("Analysis complete! ✅ Files generated: processed.csv, brand_summary.csv, sentiment_chart.png")
