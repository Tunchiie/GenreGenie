# Movie Genre Prediction - Machine Learning Project

## Background and Overview
The exponential growth of movie-related data driven by the expansion of digital media platforms has opened new opportunities to enhance user experience and recommendation systems, improving cataloging in various ways. Accurate genre prediction significantly affects the recommendation, targeted advertising, and content-based filtering used by Netflix, Disney+, and Amazon Prime, among multiple platforms. This project focuses on multi-label text classification using Natural Language Processing (NLP) techniques and supervised machine learning models. Since Movies can have a touch specific to multiple genres simultaneously, the final model must be able to handle multi-label outputs efficiently.

Key components of the pipeline include:
•	Preprocessing and cleaning of data obtained from TMDb
•	Feature extraction using TF-IDF Vectorization
•	Model training using OneVsRestClassifier and MultiOutputClassifier with Logistic Regression
•	Evaluation using precision, recall, F1-score, and visual performance metrics (confusion matrix, heatmap, precision-recall curve)

## Data Structure and Initial Checks
The processed dataset, based on The Movie Database (TMDb), included:
-	title: Title of the movie
-	overview: Short description (used for NLP)
-	genres: Multi-label genre field
-	vote_average: Average user rating
-	popularity: A metric based on views, votes, and people who've seen and liked or look forward to seeing the movie.
-	release_date: Release year (extracted from timestamp), among other features
### Initial Processing Steps:
-	Handled nulls in overview and genres
-	Converted genre IDs into genre and finally into binary multi-hot encoding
-	Applied TF-IDF vectorizer to the overview text
-	Split the dataset into training and testing sets (80/20 split)

## Executive Summary

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Precision%20recall%20curve%20for%20genre%20prediction.PNG)
Figure 1: Precision-Recall Curve illustrating the balance between genre detection accuracy and coverage.

- Goal: Predict movie genres using metadata and textual description (overview) to support informed content recommendations and genre tagging automation.

- Tools: Python, scikit-learn, pandas, NumPy, matplotlib, seaborn, feature extraction, modeling, and visualization.

- Modeling Techniques: TF-IDF Vectorization for transforming textual overviews, OneVsRest and MultiOutput classifiers for multi-label learning, and Logistic Regression for interpretability and performance balance.

- Outcome: Achieved a macro-averaged precision of 0.62 and recall of 0.49, indicating decent predictive power for popular genres, with room for improvement in infrequent genres.

- Insights:

  - The model struggled most with genre pairs that frequently co-occur, such as Action & Adventure or Drama & Romance, indicating genre ambiguity in overview structure.

  - Underrepresented genres like War, Music, and Western had the lowest recall, driven by limited training data.

  - Over-prediction of dominant genres (e.g., Drama) often led to decreased precision in nuanced categories.

  - Temporal analysis revealed rising trends in Action and Drama releases post-2020, aligning with global streaming adoption and the need for thrill and suspense.

  Vote distribution showed that most movies are moderately liked (6–8 average rating), but some high-quality, low-popularity "hidden gems" exist in niche genres.

- Impact: The framework provides a foundation for scalable genre tagging and enhances user experience in streaming services and recommendation systems.


## Evaluation Metrics
| Metric       | Score |
|--------------|-------|
| Precision    | 0.62  |
| Recall       | 0.49  |
| F1 Score     | 0.54  |


## Insights

#### Model Confusion
   
 ![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Missclassification%20heatmap%20for%20genre%20prediction.png)

Figure 2: Misclassification heatmap showing genre overlaps and confusion intensity.

The misclassification heatmap highlights genre overlap in genres like Action and Adventure that frequently co-occur and are often confused by the model due to shared overview structure (e.g., fast-paced sequences, heroic plots). This misclassification suggests the need for genre-aware contextual modeling. In addition, due to data imbalance, low-support genres such as Music or War are misclassified.

#### Confusion Matrix

 ![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Movie%20confusion%20matrix.PNG)
 
Figure 3: Confusion matrix illustrating correct and incorrect predictions for each genre.

The confusion matrix provides a genre-targeted breakdown of true positives, false positives, and false negatives. For instance, the model correctly identifies 'Drama' but often mistakes 'Romance' for 'Drama,' suggesting overlapping emotional tones and overview structures. It also struggles with underrepresented genres such as 'Documentary,' often defaulting to more frequent genres.

#### Genre Popularity Trends

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Yearly%20Movie%20release%20per%20Genre.PNG)

Figure 4: Yearly distribution of movies per genre between 2010 and 2023.

The yearly trend in genres shows a consistent rise in Action and Drama genre movie production, especially post-2020. This pattern may correlate with the popularity of streaming services and high-budget original productions. Meanwhile, genres like History and Western remain relatively niche over the years, with occasional spikes probably to tap into the nostalgia market.

#### Top Genres by Popularity

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/top%205%20movie%20Genres.PNG)
Figure 5a: Top 5 genres by movie count.

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Most%20popular%20movies.PNG)

Figure 5b: Most popular genres by aggregated popularity score.

Regarding frequency and popularity, Action, Drama, and Thriller dominate the dataset. However, while Action movies are more numerous, Drama tends to receive higher vote averages, indicating stronger audience appreciation and reflecting a high saturation of drama films in the dataset. This pattern suggests a potential bias toward high-volume genres that may not align with audience preference.

#### Hidden Gems by Genre

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Hidden%20Gems%20per%20genre.PNG)

Figure 6: High-rated but less popular movies across genres.

Hidden gems are identified by filtering movies with high average ratings but low popularity scores. These are often independent or foreign-language films overlooked due to limited distribution and subtitle aversion. Genres like Drama and Romance harbor the highest concentration of these films, suggesting untapped potential for recommendation systems.


#### Vote Distribution

![THIS IMAGE](https://github.com/Tunchiie/Machine-Learning/blob/5282d13dca29c8bc952b10972301e09a0db59f97/Average%20popularity%20per%20movie.PNG)

Figure 7: Average popularity (vote score) distribution among movies.

The vote distribution skews toward the 6–8 range, indicating a generally favorable but unenthusiastic audience reception. Very low or very high ratings are rare, suggesting a central tendency in user feedback and reducing the effectiveness of vote average as a significant feature.

## Recommendations

### Model Enhancements
-	Ensemble Models: Random Forest, Gradient Boosting, or LightGBM to capture non-linear relationships.
-	Deep Learning Models: Use LSTM, GRU, or transformer-based models (like BERT) for deeper text understanding.
-	Continuous Learning: Periodically retrain the model as more movie data becomes available.
-	Language Expansion: Extend the model to non-English movies by incorporating multilingual embeddings.
-	Genre Threshold Optimization: Tune thresholds for predicting individual genres rather than using uniform cutoffs.

### Data Improvements
-	Balance the dataset using SMOTE or weighted loss functions to handle genre imbalance.
-	Add Features: Include director, cast, keywords, production companies, and budget for richer model input.

### Product Integration
-	Streaming Services: Enhance genre tagging and suggestions for new or niche movies.
-	Content Discovery Tools: Build a front-end app for discovering movies by predicted genre and overview similarities with previously liked titles.
-	Search Engine Optimization (SEO): Auto-tag movies based on predicted genres to enhance discoverability.

## Caveats and Assumptions
-	Genre classification is inherently subjective and overlapping.
-	The model relies solely on the overview, which may be too short or vague for nuanced genre detection and often ambiguous for reliable pattern recognition.
-	Genre imbalance leads to poorer performance in rare classes (e.g., Fantasy, Music).
-	The average precision-recall metrics and genre-specific scores may vary considerably, particularly for infrequent genres.
________________________________________
