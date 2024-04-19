Here's the README for the provided script:

## Sentiment Analysis Script

This Python script performs sentiment analysis on input phrases using a trained Linear Support Vector Machine (SVM) classifier. The sentiment of a given phrase is predicted based on a custom dataset using the TF-IDF vectorization technique.

### Dependencies
- pandas: `pip install pandas`
- scikit-learn: `pip install scikit-learn`

### Usage

1. **Define Dataset**: The script begins by defining a custom dataset containing phrases and their corresponding sentiment labels. The dataset is represented as a list of tuples, where each tuple consists of a phrase and its sentiment label. For example:

```python
data = [
    ("Stop yelling at me!", 0),
    ("This is a dreadful mistake.", 1),
    ("I am thrilled with the results.", 2),
    ("I will do as you say.", 3),
    # Add more data here...
]
```

2. **Data Preprocessing**: The dataset is converted into a pandas DataFrame and preprocessed. Numerical sentiment labels are mapped to corresponding sentiment words using a dictionary.

3. **Splitting Data**: The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.

4. **Vectorization**: Text data is vectorized using TF-IDF vectorization, which converts text data into numerical vectors suitable for machine learning models.

5. **Model Training**: A Linear Support Vector Machine (SVM) classifier is trained on the training data to learn the mapping between input features (text vectors) and sentiment labels.

6. **Sentiment Prediction**: The trained model is used to predict the sentiment of new phrases passed as command-line arguments. If no phrases are provided, the script prompts the user to input one or more phrases.

### Running the Script

To run the script, execute it from the command line using Python. You can provide one or more phrases as command-line arguments:

```
python sentiment_analysis.py "I hate this weather." "This is the best day of my life."
```

The script will output the predicted sentiment for each phrase provided.

If no phrases are provided, the script will display a message asking for input.

### Additional Notes

- The accuracy of the trained model is evaluated using a separate testing set from the dataset.
- The performance of the sentiment analysis model may vary depending on the quality and size of the dataset, as well as the choice of machine learning algorithm and hyperparameters.