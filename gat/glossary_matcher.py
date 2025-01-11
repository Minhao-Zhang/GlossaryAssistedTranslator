"""
Glossary matching module for terminology management and lookup.

This module provides functionality for managing and searching domain-specific terminology
in a glossary, supporting both single-word and multi-word term lookups with stemming support.
"""

import pandas as pd
from typing import List, Dict
import os
import glob
from nltk.stem import PorterStemmer


class GlossaryMatcher:
    """A class for managing and searching domain-specific glossary terms."""

    def __init__(self):
        """
        Initialize the glossary matcher.

        Args:
            data: List of dictionaries containing glossary entries with Term, Definition, and Example fields
        """
        self.data = pd.DataFrame()
        self.stemmer = PorterStemmer()

    def search(self, term: str) -> pd.DataFrame:
        """
        Search for glossary entries matching the stemmed version of the term.

        Args:
            term: Term to search for in the glossary

        Returns:
            DataFrame containing matching glossary entries
        """
        stemmed_term = self.stemmer.stem(term)
        return self.data[self.data['StemmedTerm'] == stemmed_term]

    def to_csv(self, path: str):
        """
        Save the glossary database to a CSV file.

        Args:
            path: File path to save the glossary
        """
        self.data.to_csv(path, index=False)

    def search_sentence(self, sentence: str) -> pd.DataFrame:
        """
        Search a sentence for glossary terms by processing it into word segments.

        Handles both single-word and two-word terms, with special handling for
        hyphens and punctuation.

        Args:
            sentence: Input text to search for glossary terms

        Returns:
            DataFrame containing all matching glossary entries
        """
        # Clean and normalize the input text
        cleaned = sentence.lower()
        cleaned = ''.join(c if c.isalnum() or c ==
                          '-' else ' ' for c in cleaned)
        cleaned = cleaned.replace('-', ' ')

        # Split into words
        words = cleaned.split()

        # Generate single-word and two-word segments
        segments = words.copy()
        for i in range(len(words) - 1):
            segments.append(f"{words[i]} {words[i+1]}")

        # Only check the unique ones
        segments = list(set(segments))

        # Search each segment and combine results
        results = pd.concat([self.search(segment) for segment in segments])

        # Return unique results
        return results.drop_duplicates()

    def load_from_dir(self, dir_path: str = "data") -> None:
        """
        Load glossary data from CSV files in a directory.

        Args:
            dir_path: Path to directory containing glossary CSV files
        """
        all_files = glob.glob(os.path.join(dir_path, "*.csv"))
        dfs = []

        for file in all_files:
            df = pd.read_csv(file)
            df = df[["Term", "Translation", "Definition", "Example"]]
            df["Term"] = df["Term"].str.lower().str.replace(
                r'[^\w\s]', '', regex=True)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        self.data = pd.DataFrame(combined_df.to_dict("records"))
        if not self.data.empty:
            self.data['StemmedTerm'] = self.data['Term'].apply(
                self.stemmer.stem)


if __name__ == "__main__":
    # Initialize the glossary matcher
    matcher = GlossaryMatcher()
    matcher.load_from_dir()

    print("Glossary Matcher Service initialized. Type 'exit' to quit.")

    while True:
        sentence = input("\nEnter a sentence to search in glossary: ")
        if sentence.lower() == "exit":
            break

        results = matcher.search_sentence(sentence)

        if not results.empty:
            print("\nMatching glossary terms:")
            for term in results["Term"].unique():
                print(f"- {term}")
        else:
            print("No matching glossary terms found.")
