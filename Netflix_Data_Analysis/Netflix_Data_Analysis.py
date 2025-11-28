import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Netflix_Data_Analysis/netflix_titles.csv")
df.head()

df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')

# Clean date_added column properly
df['date_added'] = df['date_added'].astype(str).str.strip()  # remove leading/trailing spaces
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')  # convert safely
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

genres = df['listed_in'].str.split(', ', expand=True).stack()
plt.figure(figsize=(10,5))
genres.value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Popular Genres on Netflix")
plt.show()

df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Movies vs TV Shows on Netflix")
plt.ylabel("")
plt.show()

df.groupby('release_year')['show_id'].count().plot(figsize=(12,5))
plt.title("Content Release Trend (By Year)")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.show()

country_data = df['country'].str.split(', ', expand=True).stack().value_counts().head(10)
country_data.plot(kind='bar', figsize=(10,5))
plt.title("Top 10 Countries with Most Netflix Titles")
plt.show()
