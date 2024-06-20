 # README

## Reproducing the Solution

This script provides a solution for a text-based regression problem using Python and several popular libraries such as pandas, nltk, spacy, and sklearn. To reproduce the solution, follow these steps:

1. Install the required packages:

   ```
   pip install pandas nltk spacy sklearn
   ```

2. Download the pre-trained spaCy model:

   ```
   python -m spacy download en_core_web_sm
   ```

3. Download the necessary NLTK corpora:

   ```
   python
   import nltk
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

4. Ensure that the 'train.csv' and 'test.csv' files are in the same directory as the script.

5. Run the script using your preferred Python interpreter.

## Explanation

The script first imports the necessary libraries and loads the training and testing data using pandas. It then processes the text data using NLTK and spaCy, and removes stopwords and punctuation.

The processed text is then used to train a Ridge regression model using scikit-learn. The model is evaluated using mean squared error (MSE), and the results are printed.
