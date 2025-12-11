# Portuguese Fake News Detection — Report

## 1. Introduction
Automated fake news detection is increasingly important in combating misinformation online. In Portugal, the spread of fake news has implications in politics, health, and social discourse. This report describes our participation in the FakeNews-PT challenge.

**Objective:** Classify news articles as fake or real using textual features.  

**Dataset:** FakeNews-PT, including news articles labeled as Fake or Real.  

**Report Structure:**  
1. Dataset Description  
2. Feature Extraction  
3. Model Training & Evaluation  
4. Model Interpretation  
5. Clustering Analysis  
6. Conclusions  

---

## 2. Dataset Description

**Source and Size:** FakeNews-PT dataset with train/validation/test splits.  

**Data Fields:**  
- `Text`: news content  
- `Label`: 0 (Real) or 1 (Fake)  

**Basic Statistics:**  
- **Class distribution:** Bar chart of Fake vs Real labels → *Figure placeholder*  
- **Text length distribution:** Histogram/boxplot of word/character counts → *Figure placeholder*  

**Preprocessing Pipeline:**  
Four types of cleaning were applied:  
1. **No punctuation:** removed all punctuation, converted words to lowercase, removed extra whitespaces.  
2. **No punctuation + no Portuguese stop words:** removed punctuation, lowercased, removed whitespaces, removed Portuguese stop words (NLTK).  
3. **With punctuation:** kept punctuation, lowercased, removed whitespaces.  
4. **With punctuation + no Portuguese stop words:** kept punctuation, lowercased, removed whitespaces, removed Portuguese stop words.  

**Visualizations**
- Check the folder ./assets/dataset_comparison/

> **Note:** No manual tokenization was performed; TF-IDF was applied directly to the cleaned text, which internally handles tokenization.

---

## 3. Feature Extraction

**TF-IDF Configuration:**  
- `max_features = 5000`  
- `min_df = 10`  
- `max_df = 0.9`  
- IDF smoothing enabled  

**Application:** TF-IDF was applied directly to cleaned text.  

**Visualizations:**  
- Check the folder ./assets/visualizations/

---

## 4. Model Training & Evaluation

### 4.1 Models Implemented
- Decision Tree  
- Gaussian Naive Bayes  
- Logistic Regression (L2)  
- Logistic Regression (L1)  
- Multi-Layer Perceptron (architecture description)  

### 4.2 Hyperparameter Tuning
- 5-fold cross-validation  
- Model-specific hyperparameter grids  
- Best model selection based on macro-F1

### 4.3 Evaluation Metrics
- Accuracy, Precision, Recall, F1-score  
- ROC curve + AUC for best classifier  

**Experiment 1 (Ex1):** All models on all 4 cleaned datasets  
- Check the folder ./assets/dataset_comparison/

**Experiments 2 & 3 (Ex2 & Ex3):** Best-performing dataset only (punctuation)  
- Check the [notebook](../notebooks/punctuation/ex1.ipynb)

---

## 5. Model Interpretation

### 5.1 Logistic Regression Weights
- Top 10 features indicative of fake news → *Bar plot placeholder*  
- Top 10 features indicative of real news → *Bar plot placeholder*

### 5.2 L1 vs L2 Comparison
- Count of non-zero coefficients → *Table/Bar chart placeholder*  
- Discussion: sparsity, feature selection, when to prefer L1 vs L2

### 5.3 LIME Explanations — Logistic Regression
- Methodology: LIME (Ribeiro et al., 2016; Lundberg and Lee, 2017)  
- Selected samples: IDs 2921, 2437, 5557, 1697  
- LIME explanations → *Heatmaps/bar plots placeholders*  
- Discussion of feature contributions

### 5.4 MLP Interpretability
- LIME explanations for same samples → *Figure placeholders*  
- Permutation importance (1,000 samples) → *Bar plot placeholder*  
- Heatmap comparison LR vs MLP → *Heatmap placeholder*  
- Discussion: differences in feature attribution

**Visualizations:**  
- Check the [notebook](../notebooks/punctuation/ex2.ipynb)
- Check the folder ./assets/best_model/

---

## 6. Clustering

### 6.1 K-Means (K=5)
- Reasoning for K=5  
- Clustering on TF-IDF vectors

### 6.2 Cluster Inspection
- Top 3 documents per centroid → *Table placeholder*  
- Semantic labels assigned to clusters (politics, health, etc.)  
- Documents per cluster → *Bar chart placeholder*

### 6.3 PCA Visualizations
- 2D PCA plot colored by cluster → *Figure placeholder*  
- 2D PCA plot colored by true label → *Figure placeholder*  
- Discussion: cluster separability

**Visualizations:**  
- Check the [notebook](../notebooks/punctuation/ex3.ipynb)

---

## 7. Conclusions
- Summary of findings  
- Best-performing model and rationale  
- Strengths and weaknesses  
- Lessons learned  
- Future improvements

---

## 8. References
- The github repo containing the dataset
