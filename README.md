# Enron Email Dataset Analysis

## Problem Statement

In this project, I aim to analyze emails extracted from the [Enron Email Dataset](https://www.cs.cmu.edu/~enron/). The dataset is curated in the `data/enron` directory, with each email stored in a separate file. The dataset contains a mix of "spam" and "ham" (non-spam) emails. The goal is to employ natural language processing techniques to distinguish between spam and non-spam emails.

## Data Preprocessing

### Step 1: Preprocess the Data

The initial step involves creating a list of processed documents by removing punctuation, whitespace, and converting all words to lowercase. The function `make_word_list` reads each file, applies these preprocessing steps, and constructs a list of processed documents. The `process_folder` function iterates through the files in the specified folder (`data/enron`) and gathers the processed documents.

### Step 2: Construct a TF-IDF Document Matrix

TF-IDF (Term Frequency-Inverse Document Frequency) is used to create a numerical representation of the importance of terms within each document relative to the entire collection. The `TfidfVectorizer` from scikit-learn is employed with specific parameters such as `min_df=50` (minimum document frequency), `stop_words="english"` (filter out English stop words), and `max_df=0.8` (maximum document frequency). The resulting TF-IDF matrix, denoted as `X`, encapsulates information about word frequencies weighted by their relevance and rarity.

## Unsupervised Machine Learning

Unsupervised Machine Learning, specifically clustering algorithms, is employed for categorizing email data into spam and non-spam clusters. The rationale behind using unsupervised learning is its ability to autonomously identify similarities and groupings within the email content without the need for labeled training data.

## Model 1: KMeans Clustering

KMeans clustering is chosen for its simplicity, efficiency, and versatility. It efficiently handles large datasets and is applicable to various types of data, including text. Three cluster sizes (2, 3, and 5) are considered, and the number of documents assigned to each cluster is printed. The clusters are characterized based on the frequency of selected "spammy" and "hammy" words.

### Conclusions from KMeans Model

The KMeans model with k=3 emerges as the optimal choice for categorizing emails into spam and non-spam categories. This configuration strikes a balance between distinguishing the two main classes while avoiding excessive fragmentation. Increasing the number of clusters improves accuracy but becomes impractical from a business perspective.

## Model 2: Gaussian Mixture Model (GMM)

GMM is chosen for its flexibility in modeling email structures, soft assignment capabilities, and adaptability to varying email distributions. Similar to KMeans, GMMs with different cluster sizes (2, 3, and 5) are considered, and the number of documents in each cluster is printed. The clusters are characterized using the same methodology as KMeans.

### Conclusions from GMM Model

GMMs with k=3 are identified as the optimal choice, providing a balance between effective clustering and model simplicity. The model's adaptability to varied email distributions and its ability to handle uncertainty make it a robust solution for email categorization.

## Model 3: Spectral Clustering

Spectral Clustering is employed for its effectiveness in handling non-linear separation, adaptability to varying cluster shapes, and graph-based representation. Similar to previous models, Spectral Clustering is applied with cluster sizes 2, 3, and 5, and the number of documents in each cluster is printed. The clusters are characterized based on selected spammy and hammy words.

### Conclusion from Spectral Clustering

Spectral Clustering with k=2 is identified as the optimal model, excelling in capturing non-linear relationships and adapting to diverse cluster shapes. The graph-based representation and dimensionality reduction contribute to its effectiveness in email categorization.

## Final Conclusions

Spectral Clustering outperforms KMeans and GMMs in certain scenarios due to its ability to handle non-linear structures, adaptability to varying cluster shapes, graph-based representation, dimensionality reduction, and customizable affinity metrics. The choice of the optimal model depends on the specific characteristics of the email dataset and the desired trade-off between accuracy and practicality.