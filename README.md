MIMIC_Code_Prediction
Overview
MIMIC_Code_Prediction is a modular Python project for automatic ICD code prediction from clinical notes using bag-of-words features and interpretable logistic regression. It is designed for high transparency, reproducibility, and educational use on MIMIC-style datasets.

Features
Clean text preprocessing: lemmatization, abbreviation expansion, negation detection.

Bag-of-words and TF-IDF vectorization with n-grams and dimensionality reduction.

Multi-label ICD code prediction using logistic regression (OneVsRest approach) with hyperparameter tuning.

Precision-recall, ROC, and confusion matrix evaluation.

GUI for interactive predictions, result visualization, and confidence scoring.

Synthetic clinical dataset for immediate demo and experimentation.

Project Structure
Filename	Purpose
run_pipeline.py	Main CLI pipeline runner for data processing/training
preprocess.py	Text preprocessing and normalization functions
features.py	Feature extraction and selection utilities
train_model.py	Model training, evaluation, and saving/loading
evaluate.py	Evaluation and visualization plotting
gui.py	Tkinter-based GUI for ICD prediction interaction
synthetic_clinical_notes.csv	Sample clinical note dataset with ICD code labels
Setup Instructions
Clone/download the project files.

Create & activate a Python virtual environment.

Install required packages: pip install pandas scikit-learn nltk spacy joblib matplotlib tqdm negspacy.

Download necessary spaCy language models (e.g., python -m spacy download en_core_web_sm).

Run the main pipeline:

text
python run_pipeline.py --datapath synthetic_clinical_notes.csv --modelpath logregmodel.joblib --vectorizerpath featureextractor --mlbpath mlb.joblib
Start the GUI for interactive predictions:

text
python gui.py
Usage
Use CLI pipeline to preprocess, train, evaluate, and save ICD prediction models from CSV datasets.

Use GUI to paste or type clinical notes, predict ICD codes, and view results interactively with confidence score charts.
