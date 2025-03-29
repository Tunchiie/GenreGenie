# Movie Genre Prediction - Machine Learning Project

## Background
The rapid expansion of digital media platforms has generated vast volumes of movie-related data. Understanding and predicting genres from textual and metadata sources can enhance user experience, support recommendation systems, and improve media cataloging. Accurate genre prediction also plays a key role in recommendation engines, targeted advertising, and content-based filtering used by platforms such as Netflix, Amazon Prime, and Disney+.

## Overview
This project focuses on solving a multi-label text classification problem using Natural Language Processing (NLP) techniques and supervised machine learning models. Since movies can belong to multiple genres simultaneously, the model needed to handle multilabel outputs efficiently.
Key components of the pipeline include:
•	Preprocessing and cleaning of textual data (movie overviews)
•	Feature extraction using TF-IDF Vectorization
•	Model training using OneVsRestClassifier and MultiOutputClassifier with Logistic Regression
•	Evaluation using precision, recall, F1-score, and visual performance metrics (confusion matrix, heatmap, precision-recall curve)

## Executive Summary

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Precision%20recall%20curve%20for%20genre%20prediction.PNG)
Figure 1: Precision-Recall Curve illustrating the balance between genre detection accuracy and coverage.

-	Goal: Predict movie genres using metadata and textual description (overview)
-	Tools: Python, scikit-learn, pandas, NumPy, matplotlib, seaborn
-	Modeling Techniques: TF-IDF, Logistic Regression, OneVsRest, MultiOutput Classification
-	Outcome: Achieved a macro-averaged precision of 0.62 and recall of 0.49
-	Insights: Genre misclassifications occurred mostly between similar genres such as Action vs. Adventure
-	Impact: Foundation for genre-aware recommendation engines and classification tools

## Data Structure and Initial Checks
The dataset, based on The Movie Database (TMDb), included:
-	title: Title of the movie
-	overview: Short description (used for NLP)
-	genres: Multi-label genre field
-	vote_average: Average user rating
-	release_date: Release year (extracted from timestamp)
### Initial Processing Steps:
-	Handled nulls in overview and genres
-	Converted genre column into binary multi-hot encoding
-	Applied TF-IDF vectorizer to the cleaned overview text (max_features = 5000)
-	Split the dataset into training and testing sets (80/20 split)

## Insights

#### Model Confusion
   
 ![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Missclassification%20heatmap%20for%20genre%20prediction.png)

Figure 2: Misclassification heatmap showing genre overlaps and confusion intensity.

The misclassification heatmap highlights patterns in genre overlap. Genres like Action and Adventure frequently co-occur and are often confused by the model due to shared narrative elements (e.g., fast-paced sequences, heroic plots). This suggests the need for genre-aware contextual modeling. Low support genres such as Music or War tend to be misclassified due to data imbalance.
#### Confusion Matrix

 ![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Movie%20confusion%20matrix.PNG)
 
Figure 3: Confusion matrix illustrating correct and incorrect predictions for each genre.

The confusion matrix provides a genre-specific breakdown of true positives, false positives, and false negatives. For instance, the model correctly identifies 'Drama' in most cases but often mistakes 'Romance' for 'Drama', suggesting overlapping emotional tones and narrative structures. It also struggles with underrepresented genres such as 'Documentary', often defaulting to more frequent genres.
#### Genre Popularity Trends

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Yearly%20Movie%20release%20per%20Genre.PNG)

Figure 4: Yearly distribution of movies per genre between 2010 and 2023.

The yearly genre trend shows a consistent rise in Action and Drama genres, especially post-2020. This may correlate with the popularity of streaming services and high-budget original productions. Meanwhile, genres like History and Western remain relatively niche over the years, with occasional spikes.

#### Top Genres by Popularity

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/top%205%20movie%20Genres.PNG)
Figure 5a: Top 5 genres by movie count.

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Most%20popular%20movies.PNG)

Figure 5b: Most popular genres by aggregated popularity score.

From both frequency and popularity perspectives, Action, Drama, and Thriller dominate the dataset. However, while Action movies are more numerous, Drama tends to receive higher vote averages, indicating stronger audience appreciation. This suggests a potential bias toward high-volume genres that may not necessarily align with audience preference.

#### Hidden Gems by Genre

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Hidden%20Gems%20per%20genre.PNG)

Figure 6: High-rated but less popular movies across genres.

Hidden gems are identified by filtering movies with high average ratings but low popularity scores. These are often independent or foreign-language films with limited distribution. Genres like Drama and Romance harbor the highest concentration of these films, suggesting untapped potential for recommendation systems.


#### Vote Distribution

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Average%20popularity%20per%20movie.PNG)

Figure 7: Average popularity (vote score) distribution among movies.

The vote distribution skews toward the 6–8 range, indicating a general trend of positive audience reception. Very low or very high ratings are rare, suggesting central tendency in user feedback. This central clustering may introduce challenges in differentiating movie quality using average votes alone.

## Recommendations

### Model Enhancements
-	Ensemble Models: Random Forest, Gradient Boosting, or LightGBM to capture non-linear relationships.
-	Deep Learning Models: Use LSTM, GRU, or transformer-based models (like BERT) for deeper text understanding.
-	Genre Threshold Optimization: Tune thresholds for predicting individual genres rather than using uniform cutoffs.

### Data Improvements
-	Balance the dataset using SMOTE or weighted loss functions to handle genre imbalance.
-	Add Features: Include director, cast, keywords, production companies, and budget for richer model input.

### Product Integration
-	Streaming Services: Enhance genre tagging and suggestions for new or niche movies.
-	Content Discovery Tools: Build a front-end app for discovering movies by predicted genre.
-	Search Engine Optimization (SEO): Auto-tag movies based on predicted genres to enhance discoverability.

## Caveats and Assumptions
-	Genre classification is inherently subjective and overlapping.
-	The model relies solely on the overview, which may be too short or vague for nuanced genre detection.
-	Genre imbalance leads to poorer performance on rare classes (e.g., Fantasy, Music).
-	Precision-recall metrics were averaged; genre-specific scores may vary considerably.

## Additional Enhancements
-	Genre Co-occurrence Network: Visualize which genres frequently occur together to improve understanding of multi-label patterns.
-	Interactive Dashboard: Use Power BI, Streamlit, or Dash to allow users to explore predictions and insights.
-	Continuous Learning: Periodically retrain the model as more movie data becomes available.
-	Language Expansion: Extend model to non-English movies by incorporating multilingual embeddings.
________________________________________

