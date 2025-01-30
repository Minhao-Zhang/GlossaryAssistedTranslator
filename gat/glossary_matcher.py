"""
Glossary matching module for terminology management and lookup.

This module provides functionality for managing and searching domain-specific terminology
in a glossary, supporting both single-word and multi-word term lookups with stemming support.
"""

import glob
import os
from typing import Dict, List

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import warnings


class GlossaryMatcher:
    """A class for managing and searching domain-specific glossary terms."""

    def __init__(self):
        """
        Initialize the glossary matcher.

        Args:
            data: List of dictionaries containing glossary entries with Term, Definition, and Example fields
        """
        self.data = pd.DataFrame()
        self.lemmatizer = None

        # Try to download required NLTK resources
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            print("NLTK resources successfully loaded")
        except Exception as e:
            warnings.warn(
                f"Failed to load NLTK resources: {str(e)}. Using simple string matching.")
            self.lemmatizer = None

    def _lemmatize_term(self, term: str) -> str:
        """Lemmatize a term and handle plural forms."""
        if self.lemmatizer is None:
            # Fallback to simple lowercase matching
            return term.lower()

        # Split into words if multi-word term
        words = term.split()
        if len(words) == 1:
            # Single word - lemmatize noun and verb forms
            lemma = self.lemmatizer.lemmatize(term.lower(), wordnet.NOUN)
            if lemma == term.lower():
                lemma = self.lemmatizer.lemmatize(term.lower(), wordnet.VERB)
            return lemma
        else:
            # Multi-word term - lemmatize each word
            return ' '.join([self._lemmatize_term(word) for word in words])

    def search(self, term: str) -> pd.DataFrame:
        """
        Search for glossary entries matching the lemmatized version of the term.

        Args:
            term: Term to search for in the glossary

        Returns:
            DataFrame containing matching glossary entries
        """
        lemmatized_term = self._lemmatize_term(term)
        return self.data[
            (self.data['LemmatizedTerm'] == lemmatized_term) |
            (self.data['LemmatizedTermPlural'] == lemmatized_term)
        ]

    def to_csv(self, path: str):
        """
        Save the glossary database to a CSV file.

        Args:
            path: File path to save the glossary
        """
        self.data.to_csv(path, index=False)

    def search_sentence(self, sentence: str) -> pd.DataFrame:
        """
        Search a sentence for glossary terms using lemmatized matching.

        Handles both single-word and two-word terms, with support for:
        - Word inflections
        - Plural forms
        - Hyphenated terms
        - Punctuation variations

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

        # Split into words and lemmatize
        words = [self._lemmatize_term(word) for word in cleaned.split()]

        # Generate single-word and two-word segments
        segments = words.copy()
        for i in range(len(words) - 1):
            segments.append(f"{words[i]} {words[i+1]}")

        # Add plural versions of single-word segments
        segments += [word + 's' for word in words if not word.endswith('s')]

        # Only check the unique ones
        segments = list(set(segments))

        # Search each segment and combine results, filtering out empty DataFrames
        search_results = [self.search(segment) for segment in segments]
        non_empty_results = [df for df in search_results if not df.empty]

        if non_empty_results:
            results = pd.concat(non_empty_results)
            # Return unique results
            return results.drop_duplicates()
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["Term", "Translation", "Definition", "Example",
                                         "LemmatizedTerm", "LemmatizedTermPlural"])

    def load_from_dir(self, dir_path: str = "data", filter_dict: Dict[str, List[str]] = None) -> None:
        """
        Load glossary data from CSV files in a directory.

        Args:
            dir_path: Path to directory containing glossary CSV files
            filter_dict: Dictionary where key is column name and value is list of values to include

        Returns:
            None
        """
        # Initialize empty DataFrame with required columns
        self.data = pd.DataFrame(columns=["Term", "Translation", "Definition", "Example",
                                          "LemmatizedTerm", "LemmatizedTermPlural"])

        # Check if directory exists
        if not os.path.exists(dir_path):
            warnings.warn(f"Glossary directory not found: {dir_path}")
            return

        # Find CSV files
        all_files = glob.glob(os.path.join(dir_path, "*.csv"))
        if not all_files:
            warnings.warn(
                f"No CSV files found in glossary directory: {dir_path}")
            return

        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)

                # Apply filtering if filter_dict is provided
                if filter_dict is not None:
                    for col_name, allowed_values in filter_dict.items():
                        if col_name in df.columns:
                            # Convert allowed values to lowercase for case-insensitive matching
                            allowed_values = [str(v).lower()
                                              for v in allowed_values]
                            # Filter rows where column value is in allowed values
                            df = df[df[col_name].astype(
                                str).str.lower().isin(allowed_values)]

                # Select and process specific columns
                df = df[["Term", "Translation", "Definition", "Example"]]
                df["Term"] = df["Term"].str.lower().str.replace(
                    r'[^\w\s]', '', regex=True)

                dfs.append(df)
            except Exception as e:
                warnings.warn(f"Error loading glossary file {file}: {str(e)}")
                continue

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            self.data = pd.DataFrame(combined_df.to_dict("records"))
            if not self.data.empty:
                # Create lemmatized versions of terms
                self.data['LemmatizedTerm'] = self.data['Term'].apply(
                    self._lemmatize_term)
                # Create plural versions of lemmatized terms
                self.data['LemmatizedTermPlural'] = self.data['LemmatizedTerm'].apply(
                    lambda x: x + 's' if not x.endswith('s') else x)
        else:
            warnings.warn("No valid glossary data loaded")


if __name__ == "__main__":
    # Initialize the glossary matcher
    matcher = GlossaryMatcher()
    filter_dict = {}
    matcher.load_from_dir("../valo-data/", filter_dict=filter_dict)

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
