from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import TruncatedSVD
import pickle

class FeatureExtractor:
    def __init__(self, max_features=10000, ngram_range=(1,3), use_tfidf=True,
                 min_df=1, max_df=1.0, k_best=5000, n_components=300):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_tfidf = use_tfidf
        self.min_df = min_df
        self.max_df = max_df
        self.k_best = k_best
        self.n_components = n_components

        self.vectorizer = None
        self.selector = None
        self.svd = None

    def fit_transform(self, corpus, y=None):
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                max_df=self.max_df,
                min_df=self.min_df
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                max_df=self.max_df,
                min_df=self.min_df
            )
        X = self.vectorizer.fit_transform(corpus)

        if y is not None and self.k_best and self.k_best < X.shape[1]:
            self.selector = SelectKBest(chi2, k=self.k_best)
            X = self.selector.fit_transform(X, y)

        if self.n_components and self.n_components < X.shape[1]:
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            X = self.svd.fit_transform(X)
        else:
            self.svd = None

        return X

    def transform(self, corpus):
        X = self.vectorizer.transform(corpus)
        if self.selector:
            X = self.selector.transform(X)
        if self.svd:
            X = self.svd.transform(X)
        return X

    def save(self, path_prefix):
        with open(f"{path_prefix}_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        if self.selector:
            with open(f"{path_prefix}_selector.pkl", "wb") as f:
                pickle.dump(self.selector, f)
        if self.svd:
            with open(f"{path_prefix}_svd.pkl", "wb") as f:
                pickle.dump(self.svd, f)

    def load(self, path_prefix):
        with open(f"{path_prefix}_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        try:
            with open(f"{path_prefix}_selector.pkl", "rb") as f:
                self.selector = pickle.load(f)
        except FileNotFoundError:
            self.selector = None
        try:
            with open(f"{path_prefix}_svd.pkl", "rb") as f:
                self.svd = pickle.load(f)
        except FileNotFoundError:
            self.svd = None
