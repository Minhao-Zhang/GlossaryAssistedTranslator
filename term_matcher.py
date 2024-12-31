import pandas as pd
from typing import List, Dict
import os
import glob
from nltk.stem import PorterStemmer


class TermMatcher:
    def __init__(self, data: List[Dict] = None):
        """Initialize the term matcher."""
        self.data = pd.DataFrame(data) if data else pd.DataFrame()
        self.stemmer = PorterStemmer()
        if not self.data.empty:
            self.data['StemmedTerm'] = self.data['Term'].apply(
                self.stemmer.stem)

    def search(self, term: str) -> pd.DataFrame:
        """Search for entries matching the stemmed version of the term."""
        stemmed_term = self.stemmer.stem(term)
        return self.data[self.data['StemmedTerm'] == stemmed_term]

    def to_csv(self, path: str):
        """Save the term matcher database to a CSV file."""
        self.data.to_csv(path, index=False)

    def search_sentence(self, sentence: str) -> pd.DataFrame:
        """Search a sentence by processing it into 1-word and 2-word segments."""
        # Remove punctuation (except hyphens), convert to lowercase, and replace hyphens with spaces
        cleaned = sentence.lower()
        cleaned = ''.join(c if c.isalnum() or c ==
                          '-' else ' ' for c in cleaned)
        cleaned = cleaned.replace('-', ' ')

        # Split into words
        words = cleaned.split()

        # Generate 1-word and 2-word segments
        segments = words.copy()
        for i in range(len(words) - 1):
            segments.append(f"{words[i]} {words[i+1]}")

        # Search each segment and combine results
        results = pd.concat([self.search(segment) for segment in segments])

        # Return unique results
        return results.drop_duplicates()

    @classmethod
    def from_rag_db(cls, rag_db_path: str = "rag_db"):
        """Load and combine all CSV files from the rag_db folder."""
        all_files = glob.glob(os.path.join(rag_db_path, "*.csv"))
        dfs = []

        for file in all_files:
            df = pd.read_csv(file)
            # Ensure we only keep the required columns
            df = df[["Term", "Definition", "Example"]]
            # Preprocess Term column: lowercase and remove punctuation
            df["Term"] = df["Term"].str.lower().str.replace(
                r'[^\w\s]', '', regex=True)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        return cls(combined_df.to_dict("records"))


if __name__ == "__main__":
    # Initialize the term matcher
    matcher = TermMatcher.from_rag_db()

    print("Term Matcher Service initialized. Type 'exit' to quit.")

    while True:
        sentence = input("\nEnter a sentence to search: ")
        if sentence.lower() == "exit":
            break

        results = matcher.search_sentence(sentence)

        if not results.empty:
            print("\nMatching terms:")
            for term in results["Term"].unique():
                print(f"- {term}")
        else:
            print("No matching terms found.")
