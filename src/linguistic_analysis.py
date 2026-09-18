#!/usr/bin/env python3
"""Linguistic Credibility Analysis Module."""

from __future__ import annotations

import re
import string
from textblob import TextBlob

SENSATIONAL_TERMS = [
    "shocking", "unbelievable", "breaking", "exclusive", "urgent", 
    "secret", "miracle", "exposed", "you won't believe", "must see", "incredible"
]

UNATTRIBUTED_CLAIMS_PHRASES = [
    "unnamed source", "anonymous official", "according to sources", 
    "experts say", "reportedly", "it is believed"
]

class LinguisticAnalyzer:
    """Analyzes text for linguistic indicators of credibility and style."""

    def analyze(self, text: str) -> dict[str, object]:
        if not text.strip():
            return self._empty_result()

        blob = TextBlob(text)
        
        # 1. Word Count
        words = [word for word in text.split() if word.strip(string.punctuation)]
        word_count = len(words)
        
        # 2. Sentence Count
        # Using a simple heuristic to avoid needing NLTK's punkt tokenizer
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = max(1, len(sentences))
        
        # 3. Average Sentence Length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0
        
        # 4. Exclamation Mark Count
        exclamation_count = text.count("!")
        
        # 5. Question Mark Count
        question_count = text.count("?")
        
        # 6. ALL-CAPS Word Count (ignoring single letter words like 'I' or 'A')
        all_caps_count = sum(1 for word in words if word.isupper() and len(word.strip(string.punctuation)) > 1)
        
        # 7. Capitalization Ratio
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            capitalization_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        else:
            capitalization_ratio = 0.0
            
        # 8. Sentiment
        sentiment_score = blob.sentiment.polarity
        if sentiment_score > 0.1:
            sentiment_label = "Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        # 9. Subjectivity
        subjectivity_score = blob.sentiment.subjectivity
        
        # 10. Sensationalism Indicators
        text_lower = text.lower()
        sensationalism_count = sum(text_lower.count(term) for term in SENSATIONAL_TERMS)
        
        # 11. Emotional Language
        # Define thresholds for emotional language based on sentiment polarity and subjectivity
        abs_sentiment = abs(sentiment_score)
        if abs_sentiment > 0.4 and subjectivity_score > 0.6:
            emotional_language = "High"
        elif abs_sentiment > 0.2 or subjectivity_score > 0.4:
            emotional_language = "Moderate"
        else:
            emotional_language = "Low"
            
        # 12. Unattributed Claims
        unattributed_claims_count = sum(text_lower.count(phrase) for phrase in UNATTRIBUTED_CLAIMS_PHRASES)

        # Calculate Score
        score = self._calculate_score(
            sensationalism_count, all_caps_count, exclamation_count, 
            subjectivity_score, unattributed_claims_count
        )

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "all_caps_count": all_caps_count,
            "capitalization_ratio": round(capitalization_ratio, 3),
            "sentiment_label": sentiment_label,
            "sentiment_score": round(sentiment_score, 2),
            "subjectivity_score": round(subjectivity_score, 2),
            "sensationalism_count": sensationalism_count,
            "emotional_language": emotional_language,
            "unattributed_claims_count": unattributed_claims_count,
            "score": score,
            "summary": self._generate_summary(sensationalism_count, sentiment_label, emotional_language)
        }

    def _empty_result(self) -> dict[str, object]:
        return {
            "word_count": 0, "sentence_count": 0, "avg_sentence_length": 0.0,
            "exclamation_count": 0, "question_count": 0, "all_caps_count": 0,
            "capitalization_ratio": 0.0, "sentiment_label": "Neutral",
            "sentiment_score": 0.0, "subjectivity_score": 0.0,
            "sensationalism_count": 0, "emotional_language": "Low",
            "unattributed_claims_count": 0, "score": 100,
            "summary": "No text provided for linguistic analysis."
        }

    def _calculate_score(self, sensationalism: int, all_caps: int, 
                         exclamations: int, subjectivity: float, unattributed: int) -> int:
        """
        Calculate a transparent linguistic credibility score (0-100).
        Starts at 100 and deducts points for features that deviate from 
        standard objective journalistic writing.
        """
        score = 100
        score -= (sensationalism * 10)
        score -= (all_caps * 2)
        score -= (exclamations * 2)
        score -= (unattributed * 10)
        
        # Penalize excessive subjectivity (> 0.6)
        if subjectivity > 0.6:
            score -= int((subjectivity - 0.6) * 50)
            
        return max(0, min(100, score))

    def _generate_summary(self, sensationalism: int, sentiment_label: str, emotional_language: str) -> str:
        sensational_desc = "limited" if sensationalism <= 1 else "moderate" if sensationalism <= 3 else "high"
        return f"Language contains {sensational_desc} sensational wording and has a {sentiment_label.lower()} sentiment with {emotional_language.lower()} emotional language."
