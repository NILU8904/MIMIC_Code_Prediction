import pandas as pd
import argparse
from preprocess import preprocess_text
from features import FeatureExtractor
from train_model import train_logistic_regression, evaluate_model, save_model
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

def load_data(path):
    df = pd.read_csv(path)
    df['icd_codes'] = df['icd_codes'].apply(lambda x: x.split(','))
    return df

def main(args):
    print("Loading data...")
    df = load_data(args.data_path)

    print("Preprocessing text...")
    df['clean_text'] = df['clinical_note'].apply(preprocess_text)

    print("Binarizing ICD codes...")
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['icd_codes'])

    print("Extracting features...")
    fe = FeatureExtractor(max_features=10000, ngram_range=(1,3), use_tfidf=True)
    X = fe.fit_transform(df['clean_text'], y)

    print("Training model...")
    model = train_logistic_regression(X, y)

    print("Evaluating model...")
    evaluate_model(model, X, y, mlb)

    print("Saving model and vectorizer...")
    save_model(model, args.model_path)
    fe.save(args.vectorizer_path)
    joblib.dump(mlb, args.mlb_path)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICD Coding Pipeline")
    parser.add_argument('--data_path', type=str, default='synthetic_clinical_notes.csv', help='Path to dataset CSV')
    parser.add_argument('--model_path', type=str, default='logreg_model.joblib', help='Path to save trained model')
    parser.add_argument('--vectorizer_path', type=str, default='feature_extractor', help='Prefix path to save vectorizer files')
    parser.add_argument('--mlb_path', type=str, default='mlb.joblib', help='Path to save MultiLabelBinarizer')
    args = parser.parse_args()
    main(args)
