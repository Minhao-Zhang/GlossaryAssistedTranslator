"""
Glossary corrector module for managing term lists.

This module provides functionality for loading and managing a list of terms
from CSV files in a directory.
"""

import glob
import json
import os
import re


class GlossaryCorrector:
    """A class for managing a list of terms."""

    def __init__(self):
        """
        Initialize the glossary corrector.

        Creates an empty list to store terms.
        """
        self.terms = {}

    def load_from_dir(self, dir_path: str = "data") -> None:
        """
        Load terms from JSON files in a directory.

        Args:
            dir_path: Path to directory containing glossary JSON files

        Returns:
            None
        """
        # Check if directory exists
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            return

        # Find JSON files
        all_files = glob.glob(os.path.join(dir_path, "*.json"))
        if not all_files:
            print(f"No JSON files found in directory: {dir_path}")
            return

        for file in all_files:
            try:
                # Read the JSON file
                with open(file, 'r') as f:
                    data = json.load(f)

                # Extract the key-value pairs and store them in the dictionary
                for key, value in data.items():
                    if isinstance(value, list) and all(isinstance(item, str) for item in value):
                        self.terms[key] = value
                    else:
                        print(
                            f"Invalid data format in file {file}. Expected a list of strings.")

            except Exception as e:
                print(f"Error loading file {file}: {str(e)}")
                continue

    def correct_misspelling(self, sentences: list[str]) -> list[str]:
        """
        Replace glossary terms in sentences according to loaded glossary.

        This function takes a list of sentences and replaces any occurrences of
        glossary terms with their corresponding keys. The replacements are case-insensitive
        and preserve the original casing of the matched text.

        Args:
            sentences: List of input sentences to process

        Returns:
            List of sentences with glossary terms replaced
        """
        # Create list of replacement pairs sorted by descending value length
        replacements = []
        for key, values in self.terms.items():
            for value in values:
                # Create regex pattern that matches the term with any casing
                # and handles surrounding punctuation
                pattern = r'\b' + re.escape(value) + r'\b'
                replacements.append((pattern, key, value))

        # Sort by longest values first to prevent partial replacements
        replacements.sort(key=lambda x: -len(x[2]))

        # Apply replacements to each sentence
        corrected = []
        for sentence in sentences:
            modified = sentence
            for pattern, key, original_value in replacements:
                # Use a callback function to preserve original casing
                def replace_match(match):
                    matched_text = match.group()
                    # Preserve the original casing of the matched text
                    if matched_text.isupper():
                        return key.upper()
                    elif matched_text.istitle():
                        return key.title()
                    else:
                        return key.lower()

                modified = re.sub(
                    pattern,
                    replace_match,
                    modified,
                    flags=re.IGNORECASE
                )
            corrected.append(modified)

        return corrected


if __name__ == "__main__":
    # Create GlossaryCorrector instance
    corrector = GlossaryCorrector()

    # Load terms from default directory
    corrector.load_from_dir("../valo-data")

    # Test sentences
    while True:
        sentence = input(
            "\nEnter a sentence to be replaced with correct glossary: ")
        if sentence.lower() == "exit":
            break

        results = corrector.correct_misspelling([sentence])

        if results:
            print("\nCorrected sentence:")
            print(results[0])
        else:
            print("No matching glossary terms found.")
