# Portuguese Fake News Detection — Report Outline
## 1. Introduction
- Brief overview of misinformation and fake news in Portugal
- Importance of automated fake news detection
- Objective of the challenge
- Dataset summary (FakeNews-PT)
- Structure of the report

## 2. Dataset Description
- Source and size of the dataset
- Train/validation/test splits
- Data fields: `Text`, `Label`
- Basic exploratory statistics:
    - **Class distribution**: Bar chart of Fake vs Real labels
    - **Text length distribution**: Boxplot or histogram of word/character counts
- Preprocessing pipeline:
  - **Four types of cleaning were applied:**
    1. **No punctuation**: removed all punctuation, converted words to lowercase, removed extra whitespaces.
    2. **No punctuation and no Portuguese stop words**: removed punctuation, converted to lowercase, removed extra whitespaces, removed Portuguese stop words using NLTK.
    3. **With punctuation**: kept punctuation, converted to lowercase, removed extra whitespaces.
    4. **With punctuation and no Portuguese stop words**: kept punctuation, converted to lowercase, removed extra whitespaces, removed Portuguese stop words using NLTK.
- **No manual tokenization was performed**; TF-IDF was applied directly to the cleaned text, which internally handles tokenization.

## 3. Feature Extraction
- TF-IDF configuration:
  - `max_features = 5000`
  - `min_df = 10`
  - `max_df = 0.9`
  - IDF smoothing enabled
- TF-IDF applied directly to cleaned text
- Explanation of why TF-IDF is appropriate for textual classification
- Visualizations:
    - **Word clouds**: most frequent terms per class (Fake/Real)
    - **TF-IDF heatmap**: sample of terms vs documents

## 4. Model Training & Evaluation
### 4.1 Models Implemented
- Decision Tree
- Gaussian Naive Bayes
- Logistic Regression (L2)
- Logistic Regression (L1)
- Multi-Layer Perceptron (architecture description)

### 4.2 Hyperparameter Tuning
- 5-fold cross-validation methodology
- Hyperparameter grids used for each model
- Selection criteria for best models

### 4.3 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC curve + AUC for best classifier
- **Experiment 1 (Ex1)**: Run all models with all 4 cleaned datasets
  - Comparison table summarizing all models and all datasets: Table placeholder
  - Heatmap of metrics across models and datasets: Heatmap placeholder
  - Average metrics per dataset → Grouped bar chart placeholder
- **Experiments 2 & 3 (Ex2 & Ex3)**: continue only with the best-performing dataset -> check assets folder
  - Comparison table of datasets
  - Bar charts for average metrics per dataset

## 5. Model Interpretation
### 5.1 Logistic Regression Weights
- Extracting top 10 features for fake news
- Extracting top 10 features for real news
- Bar plots description

### 5.2 L1 vs L2 Comparison
- Count of non-zero coefficients in each
- Discussion on sparsity and feature selection
- When to prefer L1 vs L2 in text classification

### 5.3 LIME Explanations — Logistic Regression
- Explanation methodology
- Selected samples: IDs 2921, 2437, 5557, 1697
- Discussion of LIME outputs

### 5.4 MLP Interpretability
- LIME explanations for same selected samples
- Permutation importance (using 1000 samples)
- Comparison between MLP and Logistic Regression explanations
- Discussion of differences

## 6. Clustering
### 6.1 K-Means (K=5)
- Description and reasoning for K=5
- Clustering on TF-IDF vectors

### 6.2 Cluster Inspection
- Top 3 documents closest to each centroid
- Semantic labels assigned to each cluster (examples: politics, health, etc.)
- Documents per cluster: Bar chart placeholder

### 6.3 PCA Visualizations
- 2D PCA plot colored by cluster
- 2D PCA plot colored by true label
- Discussion of cluster separability

## 7. Conclusions
- Summary of findings
- Best-performing model and why
- Strengths and weaknesses of the approach
- Lessons learned
- Future improvements

## 8. References
- Papers, textbooks, tutorials used
- Proper citation formatting