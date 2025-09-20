import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import WordNetLemmatizer
from negspacy.negation import Negex
from negspacy.termsets import termset
import nltk

nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

# Load negation termset for clinical text
ts = termset("en_clinical")

# Register negex in pipeline
nlp.add_pipe(
    "negex",
    config={
        "ent_types": ["CONDITION"],
        "neg_termset": ts.get_patterns(),
        "extension_name": "negex",
        "chunk_prefix": ["no", "without", "denies"]
    },
    last=True
)

lemmatizer = WordNetLemmatizer()

abbrev_dict = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "copd": "chronic obstructive pulmonary disease",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident"
}

def expand_abbreviations(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbrev_dict.keys()) + r')\b')
    return pattern.sub(lambda x: abbrev_dict[x.group()], text)

def clean_text(text):
    text = text.lower()
    text = expand_abbreviations(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space or token.text in STOP_WORDS:
            continue
        lemma = lemmatizer.lemmatize(token.text)
        if hasattr(token._, "negex") and token._.negex:
            tokens.append("NOT_" + lemma)
        else:
            tokens.append(lemma)
    return " ".join(tokens)

if __name__ == "__main__":
    sample_text = "Patient has HTN and no chest pain but complains of shortness of breath."
    print("Original:", sample_text)
    print("Preprocessed:", preprocess_text(sample_text))
