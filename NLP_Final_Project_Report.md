# Fake News Style-Risk Detector: A Responsible Machine Learning Approach

**Authors:**
- **Rohaan Ahmed Khan** (ID: 6112)
- **Ammad Khan** (ID: 6013)
- **Muhammad Masood Sheikh** (ID: 6079)
- **Ibadullah Khan** (ID: 6034)

---

## Abstract
Fake-news detection is often presented as if a text classifier can decide whether an article is true or false. In reality, a classifier cannot verify factual truth without external evidence. This report presents an end-to-end, honest text-classification workflow for fake news detection. A TF-IDF and Logistic Regression pipeline is proposed as a style-risk detector. The system demonstrates a responsible approach to text classification, featuring leakage analysis, leakage-controlled training, and decision thresholds with uncertainty bands. Through feature importance analysis, the model's behavior is explained, ensuring transparency in its classification decisions.

---

## 1. Problem Definition & Literature Review

### Problem Statement
The objective of this project is to develop an NLP pipeline capable of classifying news articles as real or fake based on their textual content. Specifically, we explicitly identify and handle dataset leakage and source-confounding issues. The model must act as a style-risk detector rather than a factual truth verifier.

### Importance of the Problem
The proliferation of fake news on digital platforms poses a significant threat to public discourse, political stability, and societal trust. Automating its detection safely and responsibly is crucial. Misrepresenting a model as an objective truth arbiter is fundamentally flawed and dangerous, as AI models primarily learn stylistic differences and source artifacts rather than verifying factual claims.

### Target Users and Application Area
- **Target Users:** Journalists, content moderators, and general consumers of digital news.
- **Application Area:** Decision-support tools for flagging suspicious news content for human review.

### Literature Review
Recent research in automated fake news detection highlights the limitations of text-only classification:
1. **Conroy et al. (2015)** highlight the use of Bag-of-Words and traditional machine learning for deception detection [1].
2. **Kaliyar et al. (2020)** introduce advanced techniques leveraging contextual embeddings like BERT for fake news [2].
3. **Schuster et al. (2019)** note that models often exploit dataset-specific shortcuts (such as publisher names or specific datelines) rather than learning linguistic markers of deception [3].
4. **Ribeiro et al. (2016)** establish the importance of Explainable AI (XAI) techniques (like LIME and SHAP) to build trust in classifier predictions [4].
5. **Gravanis et al. (2019)** discuss feature engineering for fake news, emphasizing the power of combined TF-IDF and stylistic features.
6. **Bozarth et al. (2020)** analyze linguistic artifacts and show that simplistic models fail in cross-domain detection.
7. **Pérez-Rosas et al. (2018)** created a benchmark fake news dataset and showed linear models can achieve high intra-domain accuracy but suffer inter-domain.
8. **Torabi Asr et al. (2019)** showed the limitations of existing datasets which are heavily confounded by source.

### Proposed Approach
Our approach builds upon these findings by proactively identifying source confounding and emphasizing model transparency. We propose a TF-IDF vectorization coupled with Logistic Regression, explicitly stripping obvious source artifacts to force the model to learn stylistic deception markers, backed by feature-importance XAI.

---

## 2. Dataset Collection & Understanding

- **Dataset Source:** The dataset consists of two sources combined into a standard fake news benchmark: `True.csv` and `Fake.csv`.
- **Number of Records:** 1,998 total records initially, reduced to 1,991 after deduplication (992 REAL, 999 FAKE).
- **Important Features:** `title`, `text` (primary feature for classification), `subject`, and `date`.
- **Target Variable:** The target is implicitly defined by the source files (`REAL` for `True.csv` and `FAKE` for `Fake.csv`).
- **Data Quality & Characteristics:** The dataset is highly confounded by source. Most REAL articles originate from Reuters (indicated by specific datelines), while FAKE articles come from alternative sources. A "contains Reuters" heuristic alone reaches around 99.5% accuracy.
- **Ethical & Privacy Considerations:** Identifying publishers as implicitly "Fake" requires caution to avoid defamation. The data is treated purely for educational modeling.

---

## 3. Data Preprocessing & NLP Pipeline

To prepare the text for the machine learning model, a robust preprocessing pipeline is embedded via a custom `TextCleaner` class. This ensures identical treatment during both training and inference.

- **Lowercasing:** All text is converted to lowercase to ensure uniformity.
- **Text Cleaning:** Removing URLs, HTML tags, special characters, and punctuation to reduce noise.
- **Stop-word Removal:** Eliminating common but uninformative words (e.g., "the", "and") which do not contribute to style classification.
- **Handling Missing Data & Duplicates:** The pipeline strictly checks for null values and drops duplicate text entries to prevent data leakage between train and validation splits.
- **Leakage Control (Specific Step):** We implemented `--strip-source-artifacts` which removes specific datelines (like "(Reuters) -") so the model does not exploit these obvious shortcuts.

---

## 4. Feature Engineering / Text Representation

**TF-IDF Vectorization** with word n-grams (unigrams and bigrams) was implemented for text representation.

- **Suitability:** TF-IDF effectively balances the frequency of a term in a document against its frequency across the entire corpus. This penalizes overly common words and highlights distinctive stylistic markers inherent to a specific class (REAL vs FAKE).
- **N-grams:** Including unigrams and bigrams allows the model to capture short contextual phrases (e.g., "breaking news", "white house") rather than just isolated words, providing a stronger structural representation of the text.

---

## 5. NLP Model Development

We developed a pipeline using **Logistic Regression**.

- **Algorithm Explanation:** Logistic Regression is a linear classifier that computes the probability of a class based on a weighted sum of input features. It is highly interpretable, making it ideal for our requirement of explaining feature importance.
- **Hyperparameters:** Standard regularization (L2) with `C=1.0` was applied to prevent overfitting on the high-dimensional TF-IDF space.
- **Train/Validation/Test Split:** The deduplicated dataset was divided into training and holdout test sets using a stratified split (e.g., 80-20) to maintain class balance.
- **Experimental Procedure:** Cross-validation was performed on the training set to tune parameters. The pipeline encapsulates both the `TextCleaner` and the classifier, ensuring no data leakage between splits.

---

## 6. Model Evaluation & Comparison

We selected evaluation metrics appropriate for our classification problem:

- **Accuracy:** Overall correctness across both classes.
- **Precision (FAKE):** Quality of the FAKE predictions (how many predicted FAKE were actually FAKE).
- **Recall (FAKE):** Coverage of FAKE cases (how many actual FAKE were found).
- **F1-score (Macro):** Harmonic mean of precision and recall, balancing across the REAL and FAKE classes.
- **ROC-AUC:** Ranking quality across thresholds.
- **Confusion Matrix:** Shows the true positives, false positives, true negatives, and false negatives.

**Interpretation of Results:**
- **Default (Leaked):** Accuracy = 1.000, Macro F1 = 1.000, ROC-AUC = 1.000.
- **De-leaked (Artifacts stripped):** Accuracy = 0.995, Macro F1 = 0.995, ROC-AUC = 1.000.
These near-perfect scores emphasize that the model is primarily identifying stylistic and source differences rather than possessing a true understanding of factual accuracy.

---

## 7. Explainable AI (XAI)

### A. What is Explainable AI?
- **Definition:** Explainable AI (XAI) refers to methods and techniques in artificial intelligence such that the results of the solution can be understood by human experts.
- **Importance:** It contrasts with "black box" models where designers cannot explain why the AI arrived at a specific decision.
- **Interpretability vs. Explainability:** Interpretability implies the model architecture itself is understandable (like Logistic Regression weights), while explainability often refers to post-hoc tools (like SHAP) applied to complex models.
- **Importance in NLP:** Explainability is crucial to build trust, identify biases (like source confounding in our dataset), and ensure the model relies on meaningful linguistic features rather than artifacts.
- **Risks of unexplained decisions:** Treating black-box classifiers as truth arbiters can lead to unfair censorship, algorithmic bias, and propagation of systemic errors.

### B. Explain their model's decisions
Given that we used Logistic Regression over TF-IDF features, we used **Feature Importance** directly extracted from the model's learned coefficients.

**Explanation:**
- **Positive Contributions (Real News):** Words associated with journalistic conventions (e.g., "said", "tuesday", "reported") have strong positive coefficients.
- **Negative Contributions (Fake News):** Words characterized by sensationalism or opinion (e.g., "hilarious", "breaking", "watch") have strong negative coefficients.

**XAI Demonstration:**
*Input:* "Breaking: Hilarious video shows politician completely failing"
*Prediction:* Negative (FAKE)
*Explanation:* 
- "breaking" → strong negative contribution
- "hilarious" → strong negative contribution
- "failing" → contributes toward negative classification

**XAI Analysis:**
- **Influential Tokens:** Emotive and sensational words drastically influenced the FAKE predictions, while formal dateline-like words influenced REAL predictions.
- **Reasonableness:** The explanations are structurally reasonable for a style-detector.
- **Meaningful words vs Bias:** The model relied heavily on stylistic words, but prior to leakage-control, it was biased towards dateline markers (e.g., "Reuters").
- **Error Identification & Trust:** XAI proved that the initial 100% accuracy was a shortcut. By viewing feature importance, we identified the bias, stripped the artifacts, and improved structural trust in the model.
- **Limitations of Technique:** Linear feature importance only shows word-level influence and cannot capture complex, multi-word semantic reasoning or negations effectively.

---

## 8. Results, Discussion & Limitations

### Major Findings
The major finding is the extent to which dataset leakage influences NLP models. A naive implementation reports 100% accuracy and claims to have "solved" fake news. Our analysis reveals that the model essentially learned to differentiate publishers. By implementing leakage controls and providing uncertainty bands, we built a tool that is significantly more honest.

### Examples of Predictions
- **Correct predictions:** Standard wire-news correctly flagged as REAL due to formal style; highly sensational conspiracy texts flagged as FAKE.
- **Errors/Uncertainty:** Borderline probabilities (e.g., 0.45 to 0.55) are explicitly flagged as **UNCERTAIN**. Errors typically happen when a real news article uses slightly more emotive language or when a fake news article closely mimics standard journalistic formatting.

### Limitations
- **Dataset Limitations:** The dataset is highly confounded by source. It does not represent the full diversity of modern news.
- **Model Limitations:** The model cannot verify factual claims against external evidence; it only scores stylistic resemblance to the training data.
- **XAI Limitations:** TF-IDF coefficients are limited to unigram/bigram presence and ignore deep semantic context.

---

## 9. Final Report & Presentation Details

### Conclusion
This project successfully developed an NLP pipeline for fake news classification, prioritizing responsible machine learning practices. By embedding preprocessing, enforcing leakage controls, outputting uncertainty margins, and incorporating feature importance XAI, the system demonstrates how text classifiers can be used transparently as style-risk indicators rather than infallible truth detectors.

### Future Work
Future improvements include incorporating more diverse, source-balanced datasets, integrating contextual models like BERT with SHAP for deeper semantic understanding, and developing pipelines that cross-reference claims against external knowledge bases via Retrieval-Augmented Generation (RAG).

### References
1. N. J. Conroy, V. L. Rubin, and Y. Chen, "Automatic deception detection: Methods for finding fake news," *Proceedings of the Association for Information Science and Technology*, vol. 52, no. 1, pp. 1-4, 2015.
2. R. K. Kaliyar, A. Goswami, P. Narang, and S. Sinha, "FakeBERT: Fake news detection in social media with a BERT-based deep learning approach," *Multimedia Tools and Applications*, vol. 79, pp. 27765-27788, 2020.
3. T. Schuster, D. J. Shah, Y. J. S. Dar, and A. Barzilay, "Towards debiasing fact verification models," *Empirical Methods in Natural Language Processing (EMNLP)*, 2019.
4. M. T. Ribeiro, S. Singh, and C. Guestrin, '"Why should I trust you?" Explaining the predictions of any classifier,' *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 1135-1144, 2016.

### Appendix/Code Repository
The complete source code, including the `TextCleaner`, Streamlit dashboard, leakage diagnostic scripts, and model generation pipeline, is available in the accompanying repository: `Fake-News-Detector/`
