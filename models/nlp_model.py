import streamlit as st
import spacy
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from data.training_data import get_training_data

@st.cache_resource
def load_nlp_resources():
    """Load spaCy, Matcher, and trained classifier"""
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Setup Matcher for time patterns
    matcher = Matcher(nlp.vocab)
    pattern_time = [
        [{"IS_DIGIT": True}, {"TEXT": "-"}, {"IS_DIGIT": True}],
        [{"IS_DIGIT": True}, {"TEXT": ":"}, {"IS_DIGIT": True}, {"TEXT": "-"}, 
         {"IS_DIGIT": True}, {"TEXT": ":"}, {"IS_DIGIT": True}],
        [{"IS_DIGIT": True}, {"LOWER": "to"}, {"IS_DIGIT": True}],
    ]
    matcher.add("TIME", pattern_time)

    # Train classifier
    training_data = get_training_data()
    sentences, labels = zip(*training_data)
    classifier = make_pipeline(CountVectorizer(), MultinomialNB())
    classifier.fit(sentences, labels)

    return nlp, matcher, classifier