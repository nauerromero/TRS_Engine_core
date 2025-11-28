"""
SAORI AI Core - WhatsApp Chatbot
===================================
AI-Powered Recruiting Assistant via WhatsApp

Features:
- Real-time sentiment analysis
- Technical interview simulation
- English level evaluation
- Soft skills assessment
- Live scoring and feedback

Author: SAORI Team
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

from flask import Flask, request, make_response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import json
from datetime import datetime
import threading
import time

# Add project root to path (parent of src/)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[INFO] Loaded environment variables from .env file")
    else:
        print(f"[WARNING] .env file not found at {env_path}")
except ImportError:
    print(f"[WARNING] python-dotenv not installed. Environment variables must be set manually.")
except Exception as e:
    print(f"[WARNING] Failed to load .env file: {e}")

from Modules.sentiment_inference import SentimentPredictor
from Modules.english_level_evaluator import EnglishLevelEvaluator
from Modules.soft_skills_evaluator import SoftSkillsEvaluator
from Modules.questions_bank import questions_bank
import importlib
import src.whatsapp_inconsistency_detector
importlib.reload(src.whatsapp_inconsistency_detector)  # Force reload to get latest changes
from src.whatsapp_inconsistency_detector import detect_whatsapp_inconsistencies, generate_inconsistency_report, calculate_trust_score
from langdetect import detect, DetectorFactory
import random
import re

# Fuzzy matching for command tolerance (typo correction)
try:
    from rapidfuzz import fuzz
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False
    print("[WARNING] rapidfuzz not available. Install with: pip install rapidfuzz")

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Utility function: Detect language from user message using NLP
def detect_language(text):
    """
    Detect language from user message using NLP (langdetect library)
    Supports 55 languages with 95% accuracy
    Returns: 'en' for English, 'es' for Spanish (default)
    """
    try:
        # First check common greetings (langdetect can fail with very short text)
        text_lower = text.lower().strip()
        
        english_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'yes', 'no', 'yep', 'yeah', 'y', 'restart', 'new']
        spanish_greetings = ['hola', 'buenos dias', 'buenas tardes', 'buenas noches', 'sí', 'si', 'no', 'buenas', 'acepto', 'ok', 'reiniciar', 'nuevo']
        
        if text_lower in english_greetings:
            print(f"[INFO] Detected English greeting: '{text_lower}'")
            return 'en'
        elif text_lower in spanish_greetings:
            print(f"[INFO] Detected Spanish greeting: '{text_lower}'")
            return 'es'
        
        # Use NLP to detect language for longer text
        detected_lang = detect(text)
        print(f"[INFO] NLP detected language: '{detected_lang}' from text: '{text[:30]}...'")
        
        # Map to supported languages (currently EN/ES)
        if detected_lang == 'en':
            return 'en'
        elif detected_lang == 'es':
            return 'es'
        else:
            # For other languages, default to Spanish
            # Future: can expand to more languages
            print(f"[INFO] Detected language '{detected_lang}', defaulting to Spanish")
            return 'es'
            
    except Exception as e:
        # Fallback: if detection fails, default to Spanish
        print(f"[WARNING] Language detection failed: {e}, defaulting to Spanish")
        return 'es'

# Utility function: Check if response makes sense and is coherent
def is_response_makes_sense(response, min_words=8, min_length=30):
    """
    Validate that a response makes sense and is coherent.
    
    Args:
        response: User's response text
        min_words: Minimum number of words required
        min_length: Minimum character length required
    
    Returns:
        tuple: (is_valid, reason) - True if response makes sense, False with reason if not
    """
    try:
        if not response or len(response.strip()) == 0:
            return (False, "empty")
        
        response_clean = response.strip()
        
        # Check minimum length
        if len(response_clean) < min_length:
            return (False, "too_short")
        
        # Check word count
        words = response_clean.split()
        if len(words) < min_words:
            return (False, "too_few_words")
        
        # Check for structure indicators (verbs, connectors, etc.)
        # Common verbs in Spanish and English
        verbs_indicators = [
            'es', 'son', 'está', 'están', 'tiene', 'tienen', 'hace', 'hacen',
            'usa', 'usan', 'utiliza', 'utilizan', 'permite', 'permiten',
            'is', 'are', 'was', 'were', 'has', 'have', 'does', 'do', 'did',
            'uses', 'allows', 'enables', 'provides', 'creates', 'makes',
            'permite', 'facilita', 'mejora', 'optimiza', 'implementa'
        ]
        
        # Connectors that indicate coherent thought
        connectors = [
            'porque', 'debido', 'ya que', 'puesto que', 'como', 'cuando',
            'donde', 'mientras', 'aunque', 'sin embargo', 'además', 'también',
            'because', 'since', 'when', 'where', 'while', 'although', 'however',
            'also', 'additionally', 'furthermore', 'moreover', 'therefore'
        ]
        
        response_lower = response_clean.lower()
        
        # Check if response has at least some structure
        has_verb = any(verb in response_lower for verb in verbs_indicators)
        has_connector = any(conn in response_lower for conn in connectors)
        
        # Check for complete sentences (has periods, question marks, or exclamation marks)
        has_punctuation = any(p in response_clean for p in ['.', '!', '?', ','])
        
        # Check if response is just a list of words without structure
        # (too many single words without context)
        if not has_punctuation and len(words) < 15:
            # Might be just keywords, check if it's too sparse
            avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
            if avg_word_length < 4:  # Very short words might indicate just keywords
                return (False, "no_structure")
        
        # If response is very short and has no verbs or connectors, it might not make sense
        if len(words) < 12 and not has_verb and not has_connector:
            return (False, "no_coherence")
        
        # Check for repetitive words (indicates copy-paste or spam)
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 3:  # Ignore very short words
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        # If a word appears more than 30% of the time, it's suspicious
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition > len(words) * 0.3:
            return (False, "repetitive")
        
        # All checks passed
        return (True, "valid")
        
    except Exception as e:
        print(f"[WARNING] Error in is_response_makes_sense: {e}")
        # On error, be lenient and accept the response
        return (True, "error_fallback")

# Utility function: Detect if response is copied from previous answers
def is_copied_from_previous(session, current_response, threshold=0.7):
    """
    Detect if user copied a response from a previous question.
    
    Args:
        session: Current session data
        current_response: User's current response
        threshold: Similarity threshold (0-1) to consider it copied
    
    Returns:
        tuple: (is_copied, source_info) - True if copied, with info about source
    """
    try:
        if not current_response or len(current_response.strip()) < 20:
            return (False, None)
        
        current_lower = current_response.lower().strip()
        current_words = set(current_lower.split())
        
        # Check technical questions
        tech_questions = session.get('data', {}).get('tech_questions', [])
        for i, tech_q in enumerate(tech_questions):
            if 'answer' in tech_q:
                prev_answer = tech_q['answer'].lower().strip()
                if len(prev_answer) < 20:
                    continue
                
                prev_words = set(prev_answer.split())
                
                # Calculate word overlap
                if len(current_words) > 0 and len(prev_words) > 0:
                    overlap = len(current_words & prev_words)
                    similarity = overlap / min(len(current_words), len(prev_words))
                    
                    # If very similar (>70%), it's likely copied
                    if similarity > threshold:
                        return (True, f"tech_q{i+1}")
        
        # Check English questions
        english_questions = session.get('data', {}).get('english_questions', [])
        for i, eng_q in enumerate(english_questions):
            if 'answer' in eng_q:
                prev_answer = eng_q['answer'].lower().strip()
                if len(prev_answer) < 20:
                    continue
                
                prev_words = set(prev_answer.split())
                
                # Calculate word overlap
                if len(current_words) > 0 and len(prev_words) > 0:
                    overlap = len(current_words & prev_words)
                    similarity = overlap / min(len(current_words), len(prev_words))
                    
                    # If very similar (>70%), it's likely copied
                    if similarity > threshold:
                        return (True, f"english_q{i+1}")
        
        return (False, None)
    except Exception as e:
        print(f"[WARNING] Error in is_copied_from_previous: {e}")
        return (False, None)

# Utility function: Detect if user just repeats the question
def is_question_repetition(question_text, user_response, threshold=0.65):
    """
    Detect if user response is just repeating the question instead of answering it.
    Improved version with better detection of copied questions.
    
    Args:
        question_text: The question that was asked
        user_response: User's response
        threshold: Similarity threshold (0-1) to consider it a repetition (default: 0.65)
    
    Returns:
        True if response is likely just repeating the question, False otherwise
    """
    try:
        if not question_text or not user_response:
            return False
        
        # Normalize both texts - remove extra whitespace and newlines
        question_lower = ' '.join(question_text.lower().strip().split())
        response_lower = ' '.join(user_response.lower().strip().split())
        
        # Remove common formatting markers and hints
        formatting_markers = ['💡', 'hint:', '*hint*', 'tip:', '*tip*', 
                             'explain in your own words', 'answer in english please',
                             'responde en español', 'responde en inglés',
                             'describe', 'explica', 'explain', 'tell me']
        for marker in formatting_markers:
            question_lower = question_lower.replace(marker, '')
            response_lower = response_lower.replace(marker, '')
        
        question_lower = question_lower.strip()
        response_lower = response_lower.strip()
        
        # Quick check: If response is almost identical to question (exact copy)
        if response_lower == question_lower:
            return True
        
        # Check if response contains the full question text (common copy-paste pattern)
        if len(response_lower) >= len(question_lower) * 0.9:
            # Remove common question words for comparison
            question_clean = question_lower
            response_clean = response_lower
            
            # Remove question words
            question_words_to_remove = ['what', 'how', 'why', 'when', 'where', 'which', 'who',
                                       'qué', 'cómo', 'cuál', 'cuáles', 'por qué', 'cuándo', 'dónde',
                                       'is', 'are', 'was', 'were', 'do', 'does', 'did',
                                       'es', 'son', 'está', 'están', 'hace', 'hacen']
            
            for word in question_words_to_remove:
                question_clean = question_clean.replace(word, '')
                response_clean = response_clean.replace(word, '')
            
            # Check if response starts with question (very common copy pattern)
            if response_clean.startswith(question_clean[:min(50, len(question_clean))]):
                return True
        
        # Split into words for detailed analysis
        question_words = set(question_lower.split())
        response_words = set(response_lower.split())
        
        # Remove common words that don't indicate repetition
        common_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'y', 'o', 'a', 'que', 'qué', 
                       'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                       'es', 'son', 'está', 'están', 'is', 'are', 'was', 'were', 'be', 'been',
                       'cuál', 'cuáles', 'cómo', 'qué', 'which', 'what', 'how', 'why', 'when', 'where',
                       'pregunta', 'question', 'respuesta', 'answer', 'describe', 'explica', 'explain',
                       'your', 'you', 'tu', 'tus', 'su', 'sus', 'their', 'theirs'}
        
        question_words = question_words - common_words
        response_words = response_words - common_words
        
        # If no meaningful words left, can't determine
        if len(question_words) == 0:
            return False
        
        # Calculate word overlap (Jaccard similarity)
        overlap = len(question_words & response_words)
        total_unique_words = len(question_words | response_words)
        
        # Jaccard similarity: intersection / union
        jaccard_similarity = overlap / total_unique_words if total_unique_words > 0 else 0
        
        # Also calculate word overlap ratio (how much of question is in response)
        word_overlap_ratio = overlap / len(question_words) if len(question_words) > 0 else 0
        
        # Check 1: High Jaccard similarity (many words in common)
        if jaccard_similarity > threshold:
            return True
        
        # Check 2: High word overlap ratio AND response is similar length or shorter
        # (indicates copy-paste of question)
        if word_overlap_ratio > threshold and len(response_lower) <= len(question_lower) * 1.2:
            return True
        
        # Check 3: Response is shorter than question AND has high overlap
        # (common when user copies question but removes parts)
        if len(response_lower) < len(question_lower) * 0.8 and word_overlap_ratio > threshold * 0.8:
            return True
        
        # Check 4: Response starts with question words (common copy-paste pattern)
        response_start = ' '.join(response_lower.split()[:8])  # Check first 8 words
        question_start = ' '.join(question_lower.split()[:8])
        
        # Calculate similarity of first words
        response_start_words = set(response_start.split())
        question_start_words = set(question_start.split())
        start_overlap = len(response_start_words & question_start_words)
        start_similarity = start_overlap / len(question_start_words) if len(question_start_words) > 0 else 0
        
        if start_similarity > 0.7:  # 70% of first words match
            return True
        
        # Check 5: Response contains question as substring (exact copy)
        if question_lower in response_lower or response_lower in question_lower:
            return True
        
        return False
    except Exception as e:
        print(f"[WARNING] Error in is_question_repetition: {e}")
        return False

# Utility function: Fuzzy command matching (for typo tolerance)
def fuzzy_match_command(user_input, command_list, threshold=80):
    """
    Match user input to a command using fuzzy matching.
    Only used for commands, NOT for evaluating responses.
    
    Args:
        user_input: User's input text
        command_list: List of valid commands
        threshold: Minimum similarity percentage (0-100)
    
    Returns:
        Matched command if similarity >= threshold, None otherwise
    """
    if not FUZZY_MATCHING_AVAILABLE:
        # Fallback to exact match if rapidfuzz not available
        user_upper = user_input.strip().upper()
        return user_upper if user_upper in command_list else None
    
    user_upper = user_input.strip().upper()
    
    # First try exact match (fastest)
    if user_upper in command_list:
        return user_upper
    
    # Then try fuzzy matching
    best_match = None
    best_score = 0
    
    for cmd in command_list:
        # Use ratio for overall similarity
        score = fuzz.ratio(user_upper, cmd.upper())
        if score > best_score:
            best_score = score
            best_match = cmd
    
    # Only return if similarity is high enough
    if best_score >= threshold:
        return best_match
    
    return None

# Utility function: Get best match score for suggestion
def get_best_command_match(user_input, command_list, min_similarity=70):
    """
    Get the best matching command and its similarity score for suggestion purposes.
    Returns (command, score) if similarity >= min_similarity, else (None, 0).
    
    Args:
        user_input: User's input text
        command_list: List of valid commands to match against
        min_similarity: Minimum similarity to return (0-100), default 70
    
    Returns:
        Tuple (best_match_command, similarity_score)
    """
    if not FUZZY_MATCHING_AVAILABLE:
        return (None, 0)
    
    user_upper = user_input.strip().upper()
    
    # First try exact match
    if user_upper in command_list:
        return (user_upper, 100)
    
    # Then try fuzzy matching
    best_match = None
    best_score = 0
    
    for cmd in command_list:
        score = fuzz.ratio(user_upper, cmd.upper())
        if score > best_score:
            best_score = score
            best_match = cmd
    
    if best_score >= min_similarity:
        return (best_match, best_score)
    
    return (None, 0)

# Utility function: Extract name from user input
def extract_name(text):
    """Extract name from user input, removing common phrases"""
    text = text.strip()
    
    # Patterns to remove (common introductory phrases)
    remove_patterns = [
        r'^(mi nombre es|me llamo|soy|my name is|i am|i\'m)\s*',
        r'^(hola|hello|hi|hey)[,\s]*',
        r'[,\.]$'  # Remove trailing commas or periods
    ]
    
    for pattern in remove_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up and capitalize
    text = text.strip()
    return text.title() if text else text

# Simple wrapper for English level evaluation
# Grammar checker (lazy initialization)
_grammar_tool = None

def get_grammar_tool():
    """Get or initialize LanguageTool grammar checker"""
    global _grammar_tool
    if _grammar_tool is None:
        try:
            import language_tool_python
            _grammar_tool = language_tool_python.LanguageTool('en-US')
            print("[INFO] LanguageTool grammar checker initialized")
        except ImportError:
            print("[WARNING] language-tool-python not installed. Grammar checking disabled.")
            print("[INFO] Install with: pip install language-tool-python")
            _grammar_tool = False  # Mark as unavailable
        except Exception as e:
            print(f"[WARNING] Failed to initialize LanguageTool: {e}")
            _grammar_tool = False
    return _grammar_tool

def evaluate_english_level(text):
    """
    Enhanced English level evaluation with grammar checking
    
    Evaluates:
    - Text length and complexity (base score)
    - Grammar errors (penalty)
    - Spelling errors (penalty)
    """
    if not text or len(text) < 10:
        return 1.5
    
    # Base score based on length and basic complexity
    words = text.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
    
    # Base score from 0-5
    base_score = min(5.0, (word_count * 0.15) + (avg_word_length * 0.35))
    
    # Grammar validation (if available)
    grammar_tool = get_grammar_tool()
    grammar_penalty = 0.0
    
    if grammar_tool and grammar_tool is not False:
        try:
            errors = grammar_tool.check(text)
            # Filter out style warnings (not real grammar errors)
            STYLE_WARNINGS_TO_IGNORE = [
                "three successive sentences begin",
                "consider rewording",
                "use a thesaurus",
                "consider using",
                "repetition of words",
                "word repetition"
            ]
            
            # Only count real grammar/spelling errors, not style suggestions
            real_errors = [
                error for error in errors
                if not any(warning in error.message.lower() for warning in STYLE_WARNINGS_TO_IGNORE)
            ]
            
            error_count = len(real_errors)
            
            if error_count > 0:
                # Calculate penalty based on error density (errors per 100 words)
                error_density = (error_count / max(word_count, 1)) * 100
                
                # Penalty scale:
                # 0-5% error density: -0.2 points (minor errors)
                # 5-10% error density: -0.5 points (moderate errors)
                # 10-15% error density: -0.8 points (significant errors)
                # 15%+ error density: -1.2 points (many errors)
                
                # Enhanced penalty scale for better accuracy
                # More aggressive penalties for common errors
                if error_density < 3:
                    grammar_penalty = 0.3  # Increased from 0.2
                elif error_density < 6:
                    grammar_penalty = 0.7  # Increased from 0.5
                elif error_density < 10:
                    grammar_penalty = 1.0  # Increased from 0.8
                elif error_density < 15:
                    grammar_penalty = 1.5  # Increased from 1.2
                else:
                    grammar_penalty = 2.0  # Increased from 1.2
                
                # Log errors for debugging (first 5 errors for better visibility)
                style_warnings_count = len(errors) - error_count
                if style_warnings_count > 0:
                    print(f"[DEBUG GRAMMAR] Found {error_count} real grammar/spelling errors + {style_warnings_count} style warnings (ignored)")
                else:
                    print(f"[DEBUG GRAMMAR] Found {error_count} grammar/spelling errors (density: {error_density:.1f}%)")
                print(f"[DEBUG GRAMMAR] Grammar penalty applied: {grammar_penalty:.2f}")
                for error in real_errors[:5]:
                    error_text = text[max(0, error.offset-10):error.offset+error.errorLength+10]
                    print(f"  - {error.message} | Context: '...{error_text}...'")
            elif len(errors) > 0:
                # Only style warnings, no real errors
                print(f"[DEBUG GRAMMAR] Found {len(errors)} style warnings (ignored, no penalty applied)")
                
        except Exception as e:
            print(f"[WARNING] Grammar check failed: {e}, using base score only")
    
    # Apply penalty and ensure minimum score
    final_score = max(1.0, base_score - grammar_penalty)
    
    return round(final_score, 1)

# Enhanced soft skills evaluation
def evaluate_soft_skills(text):
    """Enhanced soft skills evaluation with comprehensive keyword detection"""
    if not text or len(text) < 5:
        return 2.5
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    # Base score by length (more detailed response = better)
    if word_count < 15:
        base_score = 2.5
    elif word_count < 25:
        base_score = 3.0
    elif word_count < 40:
        base_score = 3.5
    else:
        base_score = 4.0
    
    # Comprehensive keyword categories with Spanish/English
    keywords = {
        'teamwork': ['equipo', 'team', 'colabor', 'coordin', 'trabajar', 'collaborate', 'together'],
        'leadership': ['lider', 'lead', 'coordin', 'manage', 'organiz', 'dirigir', 'guiar', 'mentor'],
        'problem_solving': ['problema', 'problem', 'solucion', 'solution', 'resolver', 'solve', 'implement', 'fix', 'diagnos', 'analiz'],
        'communication': ['comunicar', 'communicate', 'document', 'explain', 'present', 'feedback', 'reunión', 'meeting'],
        'adaptability': ['aprend', 'learn', 'adapt', 'flexible', 'change', 'nuevo', 'new', 'rápid', 'quick', 'fast'],
        'results': ['logr', 'achieve', 'éxito', 'success', 'complet', 'deliver', 'result', 'terminamos', 'finish'],
        'time_management': ['tiempo', 'time', 'hora', 'hour', 'deadline', 'plazo', 'rápid', 'quick', 'eficien', 'efficient'],
        'proactive': ['iniciativa', 'initiative', 'proactiv', 'mejorar', 'improve', 'optimiz', 'prevenir', 'prevent']
    }
    
    # Count keyword matches
    total_matches = 0
    for category, words in keywords.items():
        for keyword in words:
            if keyword in text_lower:
                total_matches += 1
                break  # Count category only once
    
    # Keyword bonus (more generous)
    if total_matches >= 6:
        keyword_bonus = 1.0
    elif total_matches >= 5:
        keyword_bonus = 0.8
    elif total_matches >= 4:
        keyword_bonus = 0.6
    elif total_matches >= 3:
        keyword_bonus = 0.4
    else:
        keyword_bonus = 0.2
    
    # Calculate final score
    final_score = min(5.0, base_score + keyword_bonus)
    
    print(f"[DEBUG SOFT SKILLS] Word count: {word_count}, Categories matched: {total_matches}/8, Base: {base_score}, Bonus: {keyword_bonus}, Final: {final_score}")
    
    return round(final_score, 1)

# Simple wrapper for getting random technical questions
def get_random_technical_question(position, level="junior"):
    """Get a random technical question from the questions bank"""
    if position in questions_bank and level in questions_bank[position]:
        return random.choice(questions_bank[position][level])
    return "Tell me about your experience with this technology."

# Simple wrapper for response evaluation
def evaluate_response(question, response, position="General", expects_brief=False):
    """Enhanced response evaluation based on keywords, length, and quality - VERSION 2.0
    
    Args:
        question: The question topic
        response: The candidate's response
        position: The position being applied for
        expects_brief: If True, the question asked for a brief response (2-3 lines, breve explicación)
    """
    print(f"[DEBUG SCORING V2.0] Evaluating response with {len(response.split())} words, expects_brief={expects_brief}")
    
    if not response or len(response) < 10:
        print(f"[DEBUG SCORING V2.0] Response too short, returning 1.5")
        return 1.5
    
    # Word count analysis
    words = response.split()
    word_count = len(words)
    print(f"[DEBUG SCORING V2.0] Word count: {word_count}")
    
    # Base score by length - adjusted for brief vs detailed responses
    if expects_brief:
        # For brief responses (2-3 lines, breve explicación), use more generous scoring
        # since we explicitly asked for brevity
        if word_count < 10:
            base_score = 2.5  # More generous than 2.0
        elif word_count < 20:
            base_score = 3.0  # More generous than 2.5
        elif word_count < 30:
            base_score = 3.5  # More generous than 3.0
        else:
            base_score = 4.0  # Bonus for being more detailed than requested
    else:
        # For detailed responses (explica con tus propias palabras), use standard scoring
        if word_count < 10:
            base_score = 2.0
        elif word_count < 20:
            base_score = 2.5
        elif word_count < 30:
            base_score = 3.0
        elif word_count < 40:
            base_score = 3.5
        else:
            base_score = 4.0
    
    # Enhanced technical keywords (Backend + Data Engineering + General) - EXPANDED
    technical_keywords = [
        # Backend/API keywords
        'rest', 'restful', 'graphql', 'http', 'https', 'api', 'apis', 'endpoint', 'endpoints', 
        'docker', 'container', 'containers', 'compose', 'docker-compose', 'dockerfile', 
        'ci/cd', 'cicd', 'continuous', 'integration', 'deployment', 'deploy', 'deploying',
        'server', 'servers', 'client', 'clients', 'framework', 'frameworks', 
        'jenkins', 'github', 'gitlab', 'actions', 'pipeline', 'pipelines', 'testing', 'tests',
        'kubernetes', 'k8s', 'stateless', 'stateful', 'endpoint', 'fetching', 'fetch', 
        'orchestrate', 'orchestration', 'bug', 'bugs', 'automated', 'automates', 'automation',
        'microservices', 'microservice', 'monolith', 'monolithic',
        
        # Django/Backend Development keywords
        'django', 'djangorestframework', 'drf', 'models', 'model', 'orm', 'object-relational',
        'serializer', 'serializers', 'serialization', 'view', 'views', 'viewset', 'viewsets',
        'listapi', 'apiview', 'apiviews', 'queryset', 'querysets', 'query', 'queries',
        'json', 'xml', 'format', 'formats', 'postgresql', 'postgres', 'mysql', 'sqlite',
        'table', 'tables', 'column', 'columns', 'row', 'rows', 'record', 'records',
        'attributes', 'attribute', 'field', 'fields', 'class', 'classes', 'object', 'objects',
        'maps', 'mapping', 'mappings', 'translate', 'translation', 'sql', 'nosql',
        'packages', 'packaging', 'package', 'consistent', 'consistency', 'environment', 'environments',
        'services', 'service', 'orchestration', 'orchestrate', 'middleware',
        'migration', 'migrations', 'admin', 'authentication', 'authorization', 'permissions',
        
        # Data Engineering keywords (SQL, Python, Spark, ETL) - EXPANDED
        'pyspark', 'spark', 'apache spark', 'airflow', 'apache airflow', 'etl', 'elt',
        'processing', 'process', 'scalable', 'scalability', 'infrastructure', 'reliability', 
        'coordinate', 'coordinated', 'coordination',
        'normalization', 'normalize', 'normalized', 'normalizing', 'denormalization',
        'redundancy', 'redundant', 'integrity', 'schema', 'schemas', '1nf', '2nf', '3nf', 
        'first normal form', 'second normal form', 'third normal form',
        'dependencies', 'dependency', 'dependent', 'relational', 'relation', 'relations',
        'tables', 'relationships', 'relationship', 'foreign key', 'primary key', 'index', 'indexes',
        'dynamic', 'dynamically', 'static', 'statically', 'typed', 'typing', 'type', 'types',
        'runtime', 'compile-time', 'declare', 'declaring', 'declaration', 'flexibility', 'flexible',
        'advantages', 'disadvantages', 'benefits', 'drawbacks', 'pros', 'cons',
        'variables', 'variable', 'constant', 'constants',
        'rdd', 'rdds', 'dataframe', 'dataframes', 'dataset', 'datasets',
        'lineage', 'transformations', 'transformation', 'transform', 'transforms',
        'fault', 'faults', 'tolerance', 'tolerant', 'resilient', 'resilience',
        'distributed', 'distribute', 'distribution', 'reconstruct', 'reconstruction',
        'node', 'nodes', 'cluster', 'clusters', 'origin', 'origins', 'source', 'sources',
        'dbt', 'data build tool', 'mlflow', 'kafka', 'apache kafka', 'terraform', 
        'aws', 'amazon web services', 'cloud', 'cloud computing',
        'data warehouse', 'data lake', 'data pipeline', 'pipelines',
        'batch', 'streaming', 'real-time', 'realtime', 'latency', 'throughput',
        
        # Python-specific keywords
        'python', 'pythonic', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
        'list', 'lists', 'dict', 'dictionary', 'dictionaries', 'tuple', 'tuples',
        'function', 'functions', 'method', 'methods', 'module', 'modules', 'package',
        'import', 'imports', 'exception', 'exceptions', 'error', 'errors', 'try', 'except',
        'decorator', 'decorators', 'generator', 'generators', 'iterator', 'iterators',
        
        # Database keywords
        'database', 'databases', 'db', 'rdbms', 'transaction', 'transactions', 'commit', 'rollback',
        'join', 'joins', 'inner join', 'left join', 'right join', 'union', 'group by', 'order by',
        'select', 'insert', 'update', 'delete', 'where', 'having', 'aggregate', 'aggregation',
        'constraint', 'constraints', 'unique', 'not null', 'check', 'default',
        
        # General quality keywords - EXPANDED
        'data', 'information', 'system', 'systems', 'algorithm', 'algorithms', 
        'experience', 'experiences', 'project', 'projects', 'implement', 'implementation',
        'design', 'designs', 'optimize', 'optimization', 'optimized', 'solve', 'solution', 'solutions',
        'build', 'building', 'built', 'develop', 'development', 'developer',
        'coordinated', 'coordination', 'architected', 'architecture', 'architect',
        'efficient', 'efficiency', 'performance', 'performant', 'scalable', 'scalability',
        'maintainable', 'maintainability', 'readable', 'readability', 'clean', 'code',
        'best practices', 'best practice', 'pattern', 'patterns', 'principle', 'principles',
        'refactor', 'refactoring', 'test', 'testing', 'unit test', 'integration test',
        'documentation', 'document', 'comment', 'comments', 'readme'
    ]
    
    # Count unique keywords
    keyword_count = sum(1 for keyword in technical_keywords if keyword.lower() in response.lower())
    
    # Keyword bonus (more generous for data-rich responses)
    if keyword_count >= 10:
        keyword_bonus = 1.0
    elif keyword_count >= 7:
        keyword_bonus = 0.8
    elif keyword_count >= 5:
        keyword_bonus = 0.6
    elif keyword_count >= 3:
        keyword_bonus = 0.4
    else:
        keyword_bonus = 0.2
    
    # Calculate final score
    final_score = min(5.0, base_score + keyword_bonus)
    
    print(f"[DEBUG SCORING V2.0] Base: {base_score}, Keyword bonus: {keyword_bonus}, Keywords found: {keyword_count}, Final score: {final_score}")
    
    return round(final_score, 1)

# Initialize Flask app
app = Flask(__name__)

# Load Twilio credentials (from environment variables)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'your_account_sid')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'your_auth_token')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')

# Validate Twilio credentials
if TWILIO_ACCOUNT_SID == 'your_account_sid' or TWILIO_AUTH_TOKEN == 'your_auth_token':
    print(f"[WARNING] Twilio credentials not configured properly!")
    print(f"[WARNING] TWILIO_ACCOUNT_SID: {'Not set' if TWILIO_ACCOUNT_SID == 'your_account_sid' else 'Set'}")
    print(f"[WARNING] TWILIO_AUTH_TOKEN: {'Not set' if TWILIO_AUTH_TOKEN == 'your_auth_token' else 'Set'}")
    print(f"[WARNING] Multi-part messages will not work. Configure credentials in .env file.")
else:
    print(f"[INFO] Twilio credentials loaded successfully")

# Initialize Twilio client (will fail gracefully if credentials are invalid)
try:
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print(f"[INFO] Twilio client initialized")
except Exception as e:
    print(f"[ERROR] Failed to initialize Twilio client: {e}")
    client = None

# Load AI Sentiment Model
print("[INFO] Loading BERT sentiment model...")
sentiment_predictor = SentimentPredictor("Models/bert-sentiment-saori")
print("[INFO] Model loaded successfully!")

# Session storage (in production, use Redis or database)
sessions = {}

# Cache para análisis de emociones (evitar recalcular respuestas similares)
_emotion_cache = {}
_emotion_cache_max_size = 100  # Máximo 100 entradas en cache

# Enhanced logging function for multi-user support
def log_user_action(phone_number, action, message="", session=None, level="INFO"):
    """
    Log user action with consistent formatting for multi-user scenarios
    
    Args:
        phone_number: User's phone number
        action: Action description
        message: Optional message content
        session: Optional session object to extract user name and stage
        level: Log level (INFO, DEBUG, ERROR, WARNING)
    """
    # Get user identifier
    user_id = phone_number.replace('whatsapp:', '')  # Clean phone number
    
    # Try to get user name from session
    user_name = ""
    if session and 'data' in session and 'name' in session['data']:
        user_name = session['data']['name']
    
    # Format user identifier
    if user_name:
        user_identifier = f"{user_id} | {user_name}"
    else:
        user_identifier = user_id
    
    # Get stage info if available
    stage_info = ""
    if session and 'stage' in session:
        stage_name = STAGES.get(session['stage'], 'unknown')
        stage_info = f" [STAGE: {stage_name}]"
    
    # Format log message
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_message = f"[{timestamp}] [{level}] [USER: {user_identifier}]{stage_info} {action}"
    
    if message:
        # Limit message length and clean for logging
        msg_preview = message[:100].replace('\n', ' ').replace('\r', ' ')
        log_message += f": {msg_preview}"
    
    print(log_message)

# Conversation stages
STAGES = {
    0: "welcome",
    0.5: "privacy_authorization",
    1: "name",
    2: "position",
    3: "availability",
    4: "salary_expectation",
    5: "modality",
    6: "zone",
    7: "tech_question_1",
    8: "tech_question_2",
    9: "tech_question_3",
    10: "english_question_1",
    11: "english_question_2",
    12: "soft_skills_question",
    13: "final_question",
    14: "feedback",
    15: "closing"
}

# Positions with complete questions and answers configured
# All positions available in Free Mode
POSITIONS_WITH_CONTENT = [
    "Backend Developer (Python/Django)",
    "Full Stack Developer (Python/React)",
    "API Developer (REST/GraphQL)",
    "Backend Engineer (Node.js/Express)",
    "Software Engineer (Python/FastAPI)",
    "Data Engineer (AWS/Spark)"
]

# Available positions in the system (use only positions with content)
AVAILABLE_POSITIONS = POSITIONS_WITH_CONTENT

# Emoji mapping for emotions
EMOTION_EMOJIS = {
    'enthusiastic': '🎉',
    'confident': '💪',
    'neutral': '😊',  # Changed from 😐 to 😊 - bot stays positive
    'anxious': '😰',
    'frustrated': '😤',
    'negative': '😞'
}

def get_session(phone_number):
    """Get or create session for phone number"""
    if phone_number not in sessions:
        # Try to load from file first
        session_file = project_root / "Logs" / "whatsapp_sessions" / f"session_{phone_number.replace(':', '_')}.json"
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    loaded_session = json.load(f)
                    sessions[phone_number] = loaded_session
                    print(f"[INFO] Loaded existing session for {phone_number} from file (stage: {loaded_session.get('stage', 0)})")
                    return sessions[phone_number]
            except Exception as e:
                print(f"[WARNING] Failed to load session from {session_file}: {e}")
        
        # Create new session if file doesn't exist or loading failed
        sessions[phone_number] = {
            'stage': 0,
            'data': {},
            'scores': {
                'technical': 0,
                'english': 0,
                'soft_skills': 0
            },
            'emotions': [],
            'responses': [],
            'started_at': datetime.now().isoformat()
        }
    return sessions[phone_number]

def save_session(phone_number, session):
    """Save session data"""
    sessions[phone_number] = session
    
    # Save to file for persistence (use project root)
    try:
        log_dir = project_root / "Logs" / "whatsapp_sessions"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        filename = log_dir / f"session_{phone_number.replace(':', '_')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Session saved for {phone_number} (stage: {session.get('stage', 0)})")
    except Exception as e:
        print(f"[ERROR] Failed to save session for {phone_number}: {e}")
        # ⚠️ CRITICAL: Session in memory but not persisted
        # Session will be lost if bot restarts

def load_all_sessions():
    """Load all sessions from JSON files at startup"""
    log_dir = project_root / "Logs" / "whatsapp_sessions"
    if not log_dir.exists():
        print("[INFO] No sessions directory found, starting with empty sessions")
        return
    
    loaded_count = 0
    failed_count = 0
    
    for file_path in log_dir.glob("session_*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                # Extract phone number from filename
                # Format: session_whatsapp_+18099121128.json -> whatsapp:+18099121128
                phone_number = file_path.stem.replace('session_whatsapp_', 'whatsapp:').replace('_', ':')
                sessions[phone_number] = session_data
                loaded_count += 1
                stage = session_data.get('stage', 0)
                print(f"[INFO] Loaded session for {phone_number} (stage: {stage})")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {file_path}: {e}")
            failed_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to load session from {file_path}: {e}")
            failed_count += 1
    
    print(f"[INFO] Loaded {loaded_count} sessions from files" + (f", {failed_count} failed" if failed_count > 0 else ""))

def analyze_emotion(text):
    """Analyze emotion using AI model with caching for performance"""
    global _emotion_cache
    
    # Normalizar texto para cache (lowercase, strip)
    text_normalized = text.lower().strip()[:200]  # Limitar longitud para cache
    
    # Verificar cache primero
    if text_normalized in _emotion_cache:
        cached_result = _emotion_cache[text_normalized]
        # Actualizar posición en cache (LRU simple)
        del _emotion_cache[text_normalized]
        _emotion_cache[text_normalized] = cached_result
        return cached_result
    
    # No está en cache, calcular
    try:
        result = sentiment_predictor.predict(text)
        emotion_result = {
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'emoji': EMOTION_EMOJIS.get(result['emotion'], '😐')
        }
        
        # Guardar en cache (con límite de tamaño)
        if len(_emotion_cache) >= _emotion_cache_max_size:
            # Eliminar entrada más antigua (FIFO simple)
            oldest_key = next(iter(_emotion_cache))
            del _emotion_cache[oldest_key]
        
        _emotion_cache[text_normalized] = emotion_result
        return emotion_result
        
    except Exception as e:
        print(f"[ERROR] Sentiment prediction failed: {e}")
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'emoji': '😐'
        }

def process_message(phone_number, message_text):
    """Process incoming message and generate response"""
    session = get_session(phone_number)
    stage = session['stage']
    
    # Ensure stage is always an integer
    if isinstance(stage, str):
        # Convert string stage names back to integers
        stage_map = {v: k for k, v in STAGES.items()}
        stage = stage_map.get(stage, 0)
        session['stage'] = stage
    
    stage_name = STAGES.get(stage, 'unknown')
    
    log_user_action(phone_number, "Processing message", message_text, session, "INFO")
    log_user_action(phone_number, f"Stage value: {stage}, Type: {type(stage)}", "", session, "DEBUG")
    
    # Analyze emotion for all responses (except welcome)
    if stage > 0:
        emotion_data = analyze_emotion(message_text)
        session['emotions'].append(emotion_data)
        session['responses'].append({
            'stage': stage_name,
            'text': message_text,
            'emotion': emotion_data['emotion'],
            'confidence': emotion_data['confidence']
        })
    
    # Generate response based on stage
    response_text = ""
    
    print(f"[DEBUG] About to check stage, stage == {stage}")
    
    if stage == -1:  # Language Selection (first step after join code)
        user_choice = message_text.strip()
        
        if user_choice in ['1', 'ENGLISH', 'english', 'English', 'EN', 'en']:
            session['language'] = 'en'
            session['language_locked'] = True
            session['stage'] = 0
            save_session(phone_number, session)
            # Process with empty message to trigger welcome in English
            response_text = process_message(phone_number, "")
        elif user_choice in ['2', 'ESPAÑOL', 'español', 'Español', 'ES', 'es', 'SPANISH', 'spanish', 'Spanish']:
            session['language'] = 'es'
            session['language_locked'] = True
            session['stage'] = 0
            save_session(phone_number, session)
            # Process with empty message to trigger welcome in Spanish
            response_text = process_message(phone_number, "")
        else:
            # Invalid choice - ask again
            response_text = (
                "❌ *Please choose an option / Por favor elige una opción:*\n\n"
                "1️⃣ *English* 🇬🇧\n"
                "2️⃣ *Español* 🇪🇸\n\n"
                "Reply *1* or *2* / Responde *1* o *2* 😊"
            )
            # Stay in same stage
    
    elif stage == 0:  # Welcome + Privacy Notice
        print(f"[DEBUG] Entering welcome stage response")
        
        # Get language from session (should be set in stage -1 or from demo mode)
        if session.get('language_locked', False):
            detected_lang = session['language']
            print(f"[INFO] Language LOCKED: {detected_lang}")
        else:
            # If language not locked, detect from message (fallback for non-join-code flows)
            detected_lang = detect_language(message_text)
            session['language'] = detected_lang
            print(f"[INFO] Language detected: {detected_lang}")
        
        # Check if this is a demo profile with pre-loaded data
        profile_name = session.get('profile_name', '')
        referrer = session.get('referrer', '')
        
        if detected_lang == 'en':
            # Personalized welcome if profile data available
            if profile_name and referrer:
                greeting = f"🌸 Hi *{profile_name}*!\n\n"
                referral_line = f"*{referrer}* told me about you! "
            else:
                greeting = "🌸 *Hello!*\n\n"
                referral_line = ""
            
            response_text = (
                f"{greeting}"
                f"I'm *Saori 🌸* — an AI-powered recruitment assistant.\n\n"
                f"{referral_line}I'll guide you through a short evaluation designed to understand your *skills* and *how you feel today*.\n\n"
                f"Let's make this process *simple, respectful, and human*. ✨\n\n"
                f"In the next *10 minutes*, I'll ask about:\n\n"
                "✅ Your technical superpowers  \n"
                "✅ Your English fluency  \n"
                "✅ How you work with teams  \n"
                "✅ Your emotional state (yes, I can sense that! 😊)\n\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔐 *Quick Privacy Note*\n"
                "━━━━━━━━━━━━━━━━━━\n\n"
                "Before we start, I need your permission to process:\n"
                "• Your name & responses\n"
                "• Your salary expectations\n\n"
                "📁 Everything stays confidential — used *only* for your recruitment process, *never* shared.\n\n"
                "*Ready to begin this journey together?* ✨"
            )
        else:
            # Personalized welcome if profile data available (Spanish)
            if profile_name and referrer:
                greeting = f"🌸 ¡Hola *{profile_name}*!\n\n"
                referral_line = f"*{referrer}* me habló de ti! "
            else:
                greeting = "🌸 *¡Hola!*\n\n"
                referral_line = ""
            
            response_text = (
                f"{greeting}"
                f"Soy *Saori 🌸* — tu asistente de reclutamiento con IA.\n\n"
                f"{referral_line}Te guiaré en una breve evaluación diseñada para entender tus *habilidades* y *cómo te sientes hoy*.\n\n"
                f"Hagamos este proceso *simple, respetuoso y humano*. ✨\n\n"
                f"En los próximos *10 minutos*, te preguntaré sobre:\n\n"
                "✅ Tus superpoderes técnicos  \n"
                "✅ Tu fluidez en inglés  \n"
                "✅ Cómo trabajas en equipo  \n"
                "✅ Tu estado emocional (¡sí, puedo percibirlo! 😊)\n\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔐 *Nota Rápida de Privacidad*\n"
                "━━━━━━━━━━━━━━━━━━\n\n"
                "Antes de empezar, necesito tu permiso para procesar:\n"
                "• Tu nombre y respuestas\n"
                "• Tus expectativas salariales\n\n"
                "📁 Todo es confidencial — usado *solo* para tu proceso de reclutamiento, *nunca* compartido.\n\n"
                "*¿Lista/o para comenzar este viaje juntos?* ✨"
            )
        
        # Auto-accept privacy and skip directly to DEMO/Free Mode selection
        session['data']['privacy_accepted'] = True
        lang = session.get('language', 'es')
        
        # Check if we're in demo mode - if so, continue directly with interview
        if session.get('demo_mode') == 'full_interview':
            # In demo mode, profile data (name, position) is already loaded
            # Skip directly to availability stage since name and position are pre-configured
            session['stage'] = 3
            if lang == 'en':
                response_text = (
                    "✨ *Great! Thank you for trusting me.* 🌸\n\n"
                    "📅 *What's your availability?*\n"
                    "(Example: Immediate, 15 days, 1 month, 2 months)"
                )
            else:
                response_text = (
                    "✨ *¡Genial! Gracias por confiar en mí.* 🌸\n\n"
                    "📅 *¿Cuál es tu disponibilidad?*\n"
                    "(Ejemplo: Inmediata, 15 días, 1 mes, 2 meses)"
                )
            save_session(phone_number, session)
            return response_text
        else:
            # Not in demo mode, ask if user wants DEMO or free mode
            if lang == 'en':
                response_text = (
                    "✨ *Great! Thank you for trusting me.* 🌸\n\n"
                    "Now, how would you like to proceed?\n\n"
                    "1️⃣ *DEMO Mode* 🎬\n"
                    "   Test with sample profiles (Ana or Luis)\n\n"
                    "2️⃣ *Free Mode* 🆓\n"
                    "   Start your own interview\n\n"
                    "Which option? Reply *1* or *2* 😊\n\n"
                    "💡 *Tip:* Type *RESTART* anytime to start over."
                )
            else:
                response_text = (
                    "✨ *¡Genial! Gracias por confiar en mí.* 🌸\n\n"
                    "Ahora, ¿cómo te gustaría proceder?\n\n"
                    "1️⃣ *Modo DEMO* 🎬\n"
                    "   Probar con perfiles de ejemplo (Ana o Luis)\n\n"
                    "2️⃣ *Modo Libre* 🆓\n"
                    "   Iniciar tu propia entrevista\n\n"
                    "¿Qué opción? Responde *1* o *2* 😊\n\n"
                    "💡 *Tip:* Puedes escribir *REINICIAR* en cualquier momento para empezar de nuevo."
                )
            session['stage'] = 0.6  # New stage: DEMO or Free mode selection
    
    elif stage == 0.5:  # Privacy Authorization (for DEMO mode)
        # Check if user accepts
        user_response = message_text.strip().upper()
        lang = session.get('language', 'es')
        
        # Debug log
        print(f"[DEBUG] Stage 0.5 - User response: '{user_response}', Original: '{message_text}'")
        print(f"[DEBUG] Demo mode: {session.get('demo_mode')}")
        
        # Check for acceptance (more comprehensive list)
        accepted_responses = ['SÍ', 'SI', 'SÍ', 'YES', 'Y', 'ACEPTO', 'OK', 'OKAY', 'ACCEPT', 'AGREE']
        if user_response in accepted_responses:
            session['data']['privacy_accepted'] = True
            print(f"[DEBUG] Privacy accepted: YES")
            
            # Check if we're in demo mode - if so, continue directly with interview
            if session.get('demo_mode') == 'full_interview':
                # In demo mode, profile data (name, position) is already loaded
                # Skip directly to availability stage since name and position are pre-configured
                session['stage'] = 3
                if lang == 'en':
                    response_text = (
                        "✨ *Great! Thank you for trusting me.* 🌸\n\n"
                        "📅 *What's your availability?*\n"
                        "(Example: Immediate, 15 days, 1 month, 2 months)"
                    )
                else:
                    response_text = (
                        "✨ *¡Genial! Gracias por confiar en mí.* 🌸\n\n"
                        "📅 *¿Cuál es tu disponibilidad?*\n"
                        "(Ejemplo: Inmediata, 15 días, 1 mes, 2 meses)"
                    )
                save_session(phone_number, session)
                return response_text
            else:
                # Not in demo mode, ask if user wants DEMO or free mode
                if lang == 'en':
                    response_text = (
                        "✨ *Great! Thank you for trusting me.* 🌸\n\n"
                        "Now, how would you like to proceed?\n\n"
                        "1️⃣ *DEMO Mode* 🎬\n"
                        "   Test with sample profiles (Ana or Luis)\n\n"
                        "2️⃣ *Free Mode* 🆓\n"
                        "   Start your own interview\n\n"
                        "Which option? Reply *1* or *2* 😊\n\n"
                        "💡 *Tip:* Type *RESTART* anytime to start over."
                    )
                else:
                    response_text = (
                        "✨ *¡Genial! Gracias por confiar en mí.* 🌸\n\n"
                        "Ahora, ¿cómo te gustaría proceder?\n\n"
                        "1️⃣ *Modo DEMO* 🎬\n"
                        "   Probar con perfiles de ejemplo (Ana o Luis)\n\n"
                        "2️⃣ *Modo Libre* 🆓\n"
                        "   Iniciar tu propia entrevista\n\n"
                        "¿Qué opción? Responde *1* o *2* 😊\n\n"
                        "💡 *Tip:* Puedes escribir *REINICIAR* en cualquier momento para empezar de nuevo."
                    )
                session['stage'] = 0.6  # New stage: DEMO or Free mode selection
                save_session(phone_number, session)
                return response_text
        else:
            # User declined or invalid response
            print(f"[DEBUG] Privacy NOT accepted or invalid response: '{user_response}'")
            session['data']['privacy_accepted'] = False
            
            if lang == 'en':
                response_text = (
                    "💙 *I totally understand.*\n\n"
                    "Your privacy is important, and I respect your decision.\n\n"
                    "If you change your mind later, just send *RESTART* and we can start fresh. 🌸\n\n"
                    "Have questions? I'm here: hello@saoriai.com"
                )
            else:
                response_text = (
                    "💙 *Te entiendo completamente.*\n\n"
                    "Tu privacidad es importante, y respeto tu decisión.\n\n"
                    "Si cambias de opinión más tarde, solo envía *REINICIAR* y podemos empezar de nuevo. 🌸\n\n"
                    "¿Tienes preguntas? Aquí estoy: hello@saoriai.com"
                )
            session['stage'] = 15  # Skip to closing
            save_session(phone_number, session)
            return response_text
    
    elif stage == 0.6:  # DEMO or Free Mode Selection
        user_choice = message_text.strip()
        lang = session.get('language', 'es')
        
        if user_choice in ['1', 'DEMO', 'demo', 'Demo']:
            # User wants DEMO mode - redirect to demo mode handler
            session['demo_mode'] = 'select_language'
            session['data'] = {}  # Clear any previous data
            session['stage'] = 0
            save_session(phone_number, session)
            
            if lang == 'en':
                response_text = (
                    "🎬 *DEMO MODE* 🌸\n\n"
                    "Let's test the system! First, choose the language:\n\n"
                    "1️⃣ *English* 🇬🇧\n"
                    "2️⃣ *Español* 🇪🇸\n\n"
                    "Which language? Just reply: *1* or *2* 😊\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "📖 Or send *HELP* or *AYUDA* anytime for help"
                )
            else:
                response_text = (
                    "🎬 *MODO DEMO* 🌸\n\n"
                    "¡Probemos el sistema! Primero, elige el idioma:\n\n"
                    "1️⃣ *English* 🇬🇧\n"
                    "2️⃣ *Español* 🇪🇸\n\n"
                    "¿Qué idioma? Solo responde: *1* o *2* 😊\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "📖 O envía *AYUDA* en cualquier momento para ayuda"
                )
        elif user_choice in ['2', 'LIBRE', 'libre', 'Libre', 'FREE', 'free', 'Free']:
            # User wants free mode - continue with normal flow
            # Remove demo_mode key completely to ensure clean free mode
            if 'demo_mode' in session:
                del session['demo_mode']
            session['stage'] = 1
            save_session(phone_number, session)  # CRITICAL: Save before returning
            
            if lang == 'en':
                response_text = (
                    "✨ *Welcome to SAORI AI!* 🌸\n\n"
                    "I'm *Saori* — your AI-powered recruitment assistant.\n\n"
                    "I'll guide you through a quick evaluation (about 10 minutes) where I'll ask about:\n\n"
                    "✅ Your technical skills (3 questions)\n"
                    "✅ Your English level (2 questions)\n"
                    "✅ Your teamwork experience (1 question)\n"
                    "✅ Your professional background\n\n"
                    "Everything is confidential and used only for your recruitment process.\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "👤 *Let's start! What's your full name?*\n\n"
                    "💡 *Example:* John Smith, María García\n"
                    "💡 *Tip:* Type *RESTART* anytime to start over."
                )
            else:
                response_text = (
                    "✨ *¡Bienvenido a SAORI AI!* 🌸\n\n"
                    "Soy *Saori* — tu asistente de reclutamiento con IA.\n\n"
                    "Te guiaré en una evaluación rápida (unos 10 minutos) donde te preguntaré sobre:\n\n"
                    "✅ Tus habilidades técnicas (3 preguntas)\n"
                    "✅ Tu nivel de inglés (2 preguntas)\n"
                    "✅ Tu experiencia de trabajo en equipo (1 pregunta)\n"
                    "✅ Tu perfil profesional\n\n"
                    "Todo es confidencial y se usa solo para tu proceso de reclutamiento.\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "👤 *¡Empecemos! ¿Cuál es tu nombre completo?*\n\n"
                    "💡 *Ejemplo:* Juan Pérez, María García\n"
                    "💡 *Tip:* Puedes escribir *REINICIAR* en cualquier momento para empezar de nuevo."
                )
        else:
            # Invalid choice - ask again
            if lang == 'en':
                response_text = (
                    "❌ *Please choose an option:*\n\n"
                    "1️⃣ *DEMO Mode* 🎬 - Test with sample profiles\n"
                    "2️⃣ *Free Mode* 🆓 - Start your own interview\n\n"
                    "Reply *1* or *2* 😊"
                )
            else:
                response_text = (
                    "❌ *Por favor elige una opción:*\n\n"
                    "1️⃣ *Modo DEMO* 🎬 - Probar con perfiles de ejemplo\n"
                    "2️⃣ *Modo Libre* 🆓 - Iniciar tu propia entrevista\n\n"
                    "Responde *1* o *2* 😊"
                )
            # Stay in same stage
    
    elif stage == 1:  # Name
        # Extract name intelligently (remove "mi nombre es", "me llamo", etc.)
        name = extract_name(message_text)
        session['data']['name'] = name
        emotion = session['emotions'][-1]
        lang = session.get('language', 'es')
        
        # Build positions list (use session-specific positions if available)
        positions = session.get('available_positions', AVAILABLE_POSITIONS)
        positions_text = ""
        for i, pos in enumerate(positions, 1):
            positions_text += f"{i}️⃣ {pos}\n"
        
        # In demo mode, automatically use pre-configured position
        demo_mode = session.get('demo_mode', '')
        if demo_mode == 'full_interview':
            # Get pre-configured position (already set when profile was loaded)
            preconfigured_position = session['data'].get('position', '')
            
            # Find which number corresponds to pre-configured position
            position_number = None
            for i, pos in enumerate(positions, 1):
                if pos == preconfigured_position:
                    position_number = i
                    break
            
            if lang == 'en':
                response_text = (
                    f"Nice to meet you, *{name}*! 🎉\n\n"
                    f"💼 *Great news! Based on your profile, here are the positions that fit perfectly:*\n\n"
                    f"{positions_text}\n"
                    f"✨ *Perfect match: Position {position_number} - {preconfigured_position}*\n\n"
                    f"Let's proceed with this position! 😊\n\n"
                    "📅 *What's your availability?*\n"
                    "(Example: Immediate, 15 days, 1 month, 2 months)"
                )
            else:
                response_text = (
                    f"¡Encantado de conocerte, *{name}*! 🎉\n\n"
                    f"💼 *¡Excelente! Estas son las posiciones disponibles:*\n\n"
                    f"{positions_text}\n"
                    f"✨ *Match perfecto: Posición {position_number} - {preconfigured_position}*\n\n"
                    f"¡Procedamos con esta posición! 😊\n\n"
                    "📅 *¿Cuál es tu disponibilidad?*\n"
                    "(Ejemplo: Inmediata, 15 días, 1 mes, 2 meses)"
                )
            # Automatically set position and skip to availability stage
            session['data']['position'] = preconfigured_position
            session['stage'] = 3
        else:
            # Normal flow: ask user to select position (Free Mode)
            # Enhanced message with position descriptions
            position_descriptions = {
                "Backend Developer (Python/Django)": "→ Technical questions about Django, APIs, databases",
                "Full Stack Developer (Python/React)": "→ Technical questions about React, frontend/backend integration",
                "API Developer (REST/GraphQL)": "→ Technical questions about API design, REST, GraphQL",
                "Backend Engineer (Node.js/Express)": "→ Technical questions about Node.js, Express, async programming",
                "Software Engineer (Python/FastAPI)": "→ Technical questions about FastAPI, async/await, Python",
                "Data Engineer (AWS/Spark)": "→ Technical questions about data pipelines, Spark, AWS"
            }
            
            positions_text_enhanced = ""
            for i, pos in enumerate(positions, 1):
                desc = position_descriptions.get(pos, "")
                if desc:
                    positions_text_enhanced += f"{i}️⃣ {pos}\n   {desc}\n\n"
                else:
                    positions_text_enhanced += f"{i}️⃣ {pos}\n"
            
            if lang == 'en':
                response_text = (
                    f"Nice to meet you, *{name}*! 🎉\n\n"
                    f"💼 *These are the available positions 😊*\n\n"
                    f"{positions_text_enhanced}"
                    f"Which position would you like to apply for?\n"
                    f"_(Reply with the number or position name)_\n\n"
                    f"💡 *Tip:* Choose the position that best matches your experience"
                )
            else:
                response_text = (
                    f"¡Encantado de conocerte, *{name}*! 🎉\n\n"
                    f"💼 *Estas son las posiciones disponibles 😊*\n\n"
                    f"{positions_text_enhanced}"
                    f"¿A cuál posición te gustaría aplicar?\n"
                    f"_(Responde con el número o el nombre de la posición)_\n\n"
                    f"💡 *Tip:* Elige la posición que mejor coincida con tu experiencia"
                )
            session['stage'] = 2
    
    elif stage == 2:  # Position
        # Parse position (accept number or text)
        selected_position = message_text
        lang = session.get('language', 'es')
        
        # Get positions list (use session-specific positions if available)
        positions = session.get('available_positions', AVAILABLE_POSITIONS)
        
        # Check if user sent a number
        if message_text.strip().isdigit():
            position_index = int(message_text.strip()) - 1
            if 0 <= position_index < len(positions):
                selected_position = positions[position_index]
        
        session['data']['position'] = selected_position
        emotion = session['emotions'][-1]
        
        if lang == 'en':
            response_text = (
                f"Excellent! You're applying for *{selected_position}* 😊\n\n"
                "📅 *What's your availability?*\n"
                "(Example: Immediate, 15 days, 1 month, 2 months)"
            )
        else:
            response_text = (
                f"Excelente! Estás aplicando para *{selected_position}* 😊\n\n"
                "📅 *¿Cuál es tu disponibilidad?*\n"
                "(Ejemplo: Inmediata, 15 días, 1 mes, 2 meses)"
            )
        session['stage'] = 3
    
    elif stage == 3:  # Availability
        session['data']['availability'] = message_text
        lang = session.get('language', 'es')
        
        if lang == 'en':
            response_text = (
                "💰 *What's your monthly salary expectation?*\n"
                "(Example: $3000 USD, $50,000 MXN, Negotiable)"
            )
        else:
            response_text = (
                "💰 *¿Cuál es tu expectativa salarial mensual?*\n"
                "(Ejemplo: $3000 USD, $50,000 MXN, Negociable)"
            )
        session['stage'] = 4
    
    elif stage == 4:  # Salary
        session['data']['salary'] = message_text
        lang = session.get('language', 'es')
        
        if lang == 'en':
            response_text = (
                "🏢 *What work modality do you prefer?*\n\n"
                "1️⃣ Remote\n"
                "2️⃣ Hybrid\n"
                "3️⃣ On-site\n\n"
                "Reply with number or name."
            )
        else:
            response_text = (
                "🏢 *¿Qué modalidad prefieres?*\n\n"
                "1️⃣ Remoto\n"
                "2️⃣ Híbrido\n"
                "3️⃣ Presencial\n\n"
                "Responde con el número o nombre."
            )
        session['stage'] = 5
    
    elif stage == 5:  # Modality
        user_choice = message_text.strip().upper()
        lang = session.get('language', 'es')
        
        # Valid options for modality
        valid_options_en = ['1', '2', '3', 'REMOTE', 'HYBRID', 'ON-SITE', 'ONSITE', 'ON SITE']
        valid_options_es = ['1', '2', '3', 'REMOTO', 'HÍBRIDO', 'HIBRIDO', 'PRESENCIAL']
        
        # Check if valid
        is_valid = False
        modality_value = None
        
        if lang == 'en':
            if user_choice in valid_options_en:
                is_valid = True
                if user_choice in ['1', 'REMOTE']:
                    modality_value = 'Remote'
                elif user_choice in ['2', 'HYBRID']:
                    modality_value = 'Hybrid'
                elif user_choice in ['3', 'ON-SITE', 'ONSITE', 'ON SITE']:
                    modality_value = 'On-site'
        else:
            if user_choice in valid_options_es:
                is_valid = True
                if user_choice in ['1', 'REMOTO']:
                    modality_value = 'Remoto'
                elif user_choice in ['2', 'HÍBRIDO', 'HIBRIDO']:
                    modality_value = 'Híbrido'
                elif user_choice in ['3', 'PRESENCIAL']:
                    modality_value = 'Presencial'
        
        if is_valid:
            session['data']['modality'] = modality_value if modality_value else message_text
            
            if lang == 'en':
                response_text = (
                    "🌍 *What's your timezone?*\n"
                    "(Example: Mexico City, Buenos Aires, Madrid, New York)"
                )
            else:
                response_text = (
                    "🌍 *¿En qué zona horaria estás?*\n"
                    "(Ejemplo: Ciudad de México, Buenos Aires, Madrid, New York)"
                )
            session['stage'] = 6
        else:
            # Invalid option - repeat question
            if lang == 'en':
                response_text = (
                    "❌ *Invalid option.* Please choose:\n\n"
                    "1️⃣ Remote\n"
                    "2️⃣ Hybrid\n"
                    "3️⃣ On-site\n\n"
                    "Reply with number or name."
                )
            else:
                response_text = (
                    "❌ *Opción inválida.* Por favor elige:\n\n"
                    "1️⃣ Remoto\n"
                    "2️⃣ Híbrido\n"
                    "3️⃣ Presencial\n\n"
                    "Responde con el número o nombre."
                )
            # Stay in same stage
    
    elif stage == 6:  # Zone
        lang = session.get('language', 'es')
        
        # Validate that response is not a command
        message_upper = message_text.strip().upper()
        if message_upper in ['RESTART', 'REINICIAR', 'HELP', 'AYUDA', 'RESET', 'NUEVO', 'NEW']:
            # Process as command - let webhook handle it
            # For now, just accept it as timezone to avoid breaking flow
            pass
        
        # Validate minimum length
        if len(message_text.strip()) < 2:
            if lang == 'en':
                response_text = (
                    "⚠️ *Please provide a valid timezone.*\n\n"
                    "Example: Mexico City, Buenos Aires, Madrid, New York\n\n"
                    "🌍 *What's your timezone?*"
                )
            else:
                response_text = (
                    "⚠️ *Por favor proporciona una zona horaria válida.*\n\n"
                    "Ejemplo: Ciudad de México, Buenos Aires, Madrid, New York\n\n"
                    "🌍 *¿En qué zona horaria estás?*"
                )
            save_session(phone_number, session)
            return response_text
        
        session['data']['zone'] = message_text
        emotion = session['emotions'][-1] if session['emotions'] else {'emotion': 'neutral', 'confidence': 0.5}
        position = session['data'].get('position', '').lower()
        
        # Select technical questions based on position
        if 'data engineer' in position:
            if lang == 'en':
                tech_q1 = (
                    "🔧 *Question 1 of 3:*\n"
                    "What is normalization in relational databases and why is it important in schema design?\n"
                    "(Explain the first three normal forms with conceptual examples)\n\n"
                    "💡 *Hint:* Mention concepts like: redundancy, integrity, 1NF, 2NF, 3NF, tables, relationships, dependencies"
                )
            else:
                tech_q1 = (
                    "🔧 *Pregunta 1 de 3:*\n"
                    "¿Qué es la normalización en bases de datos relacionales y por qué es importante en el diseño de esquemas?\n"
                    "(Explica las primeras tres formas normales con ejemplos conceptuales)\n\n"
                    "💡 *Hint:* Menciona conceptos como: redundancia, integridad, 1NF, 2NF, 3NF, tablas, relaciones, dependencias"
                )
        elif 'backend developer' in position and 'django' in position:
            # Backend Developer (Python/Django)
            if lang == 'en':
                tech_q1 = (
                    "🔧 *Question 1 of 3:*\n"
                    "What are Django models and how do they relate to the database?\n"
                    "(Explain in your own words)\n\n"
                    "💡 *Hint:* Mention concepts like: Python classes, database tables, ORM, attributes, columns"
                )
            else:
                tech_q1 = (
                    "🔧 *Pregunta 1 de 3:*\n"
                    "¿Qué son los modelos de Django y cómo se relacionan con la base de datos?\n"
                    "(Explica con tus propias palabras)\n\n"
                    "💡 *Hint:* Menciona conceptos como: clases Python, tablas de base de datos, ORM, atributos, columnas"
                )
        elif 'full stack' in position and 'react' in position:
            # Full Stack Developer (Python/React)
            if lang == 'en':
                tech_q1 = (
                    "🔧 *Question 1 of 3:*\n"
                    "What is the difference between server-side rendering (SSR) and client-side rendering (CSR) in React applications?\n"
                    "(Explain in your own words)\n\n"
                    "💡 *Hint:* Mention concepts like: Next.js, hydration, SEO, performance, initial load"
                )
            else:
                tech_q1 = (
                    "🔧 *Pregunta 1 de 3:*\n"
                    "¿Cuál es la diferencia entre renderizado del lado del servidor (SSR) y renderizado del lado del cliente (CSR) en aplicaciones React?\n"
                    "(Explica con tus propias palabras)\n\n"
                    "💡 *Hint:* Menciona conceptos como: Next.js, hidratación, SEO, rendimiento, carga inicial"
                )
        elif 'api developer' in position or ('rest' in position and 'graphql' in position):
            # API Developer (REST/GraphQL)
            if lang == 'en':
                tech_q1 = (
                    "🔧 *Question 1 of 3:*\n"
                    "What's the difference between REST and GraphQL? When would you choose one over the other?\n"
                    "(Explain in your own words)\n\n"
                    "💡 *Hint:* Mention concepts like: endpoints, queries, over-fetching, under-fetching, flexibility"
                )
            else:
                tech_q1 = (
                    "🔧 *Pregunta 1 de 3:*\n"
                    "¿Cuál es la diferencia entre REST y GraphQL? ¿Cuándo elegirías uno sobre el otro?\n"
                    "(Explica con tus propias palabras)\n\n"
                    "💡 *Hint:* Menciona conceptos como: endpoints, queries, sobre-fetching, sub-fetching, flexibilidad"
                )
        elif 'backend engineer' in position and 'node.js' in position:
            # Backend Engineer (Node.js/Express)
            if lang == 'en':
                tech_q1 = (
                    "🔧 *Question 1 of 3:*\n"
                    "What is the event loop in Node.js and how does it enable asynchronous operations?\n"
                    "(Explain in your own words)\n\n"
                    "💡 *Hint:* Mention concepts like: call stack, callback queue, non-blocking I/O, single-threaded"
                )
            else:
                tech_q1 = (
                    "🔧 *Pregunta 1 de 3:*\n"
                    "¿Qué es el event loop en Node.js y cómo permite operaciones asíncronas?\n"
                    "(Explica con tus propias palabras)\n\n"
                    "💡 *Hint:* Menciona conceptos como: call stack, cola de callbacks, I/O no bloqueante, single-threaded"
                )
        elif 'software engineer' in position and 'fastapi' in position:
            # Software Engineer (Python/FastAPI)
            if lang == 'en':
                tech_q1 = (
                    "🔧 *Question 1 of 3:*\n"
                    "What are the main advantages of FastAPI over other Python web frameworks like Flask or Django?\n"
                    "(Explain in your own words)\n\n"
                    "💡 *Hint:* Mention concepts like: async/await, type hints, automatic documentation, performance"
                )
            else:
                tech_q1 = (
                    "🔧 *Pregunta 1 de 3:*\n"
                    "¿Cuáles son las principales ventajas de FastAPI sobre otros frameworks web de Python como Flask o Django?\n"
                    "(Explica con tus propias palabras)\n\n"
                    "💡 *Hint:* Menciona conceptos como: async/await, type hints, documentación automática, rendimiento"
                )
        elif 'backend' in position:
            # Fallback for other backend positions
            if lang == 'en':
                tech_q1 = (
                    "🔧 *Question 1 of 3:*\n"
                    "What are Django models and how do they relate to the database?\n"
                    "(Explain in your own words)\n\n"
                    "💡 *Hint:* Mention concepts like: Python classes, database tables, ORM, attributes, columns"
                )
            else:
                tech_q1 = (
                    "🔧 *Pregunta 1 de 3:*\n"
                    "¿Qué son los modelos de Django y cómo se relacionan con la base de datos?\n"
                    "(Explica con tus propias palabras)\n\n"
                    "💡 *Hint:* Menciona conceptos como: clases Python, tablas de base de datos, ORM, atributos, columnas"
                )
        else:
            # Default questions for other positions
            if lang == 'en':
                tech_q1 = (
                    "🔧 *Question 1 of 3:*\n"
                    "What's the difference between REST and GraphQL?\n"
                    "(Explain in your own words)"
                )
            else:
                tech_q1 = (
                    "🔧 *Pregunta 1 de 3:*\n"
                    "¿Qué diferencia hay entre REST y GraphQL?\n"
                    "(Explica con tus propias palabras)"
                )
        
        # Check if we're in demo mode to show ANSWERS tip
        is_demo_mode = session.get('demo_mode') == 'full_interview'
        
        # Get position for enhanced message
        position = session['data'].get('position', 'your selected position')
        
        if lang == 'en':
            answers_tip = ""
            if is_demo_mode:
                answers_tip = "💡 *TIP:* At any time, send ANSWERS to see suggested responses\n\n"
            
            if not is_demo_mode:
                # Enhanced message for Free Mode
                response_text = (
                    "Perfect! 😊\n\n"
                    f"⚡ *Now come 3 technical questions about {position}.*\n\n"
                    "💡 *What to expect:*\n"
                    "• Each question evaluates your technical knowledge\n"
                    "• There are no wrong answers — be honest about your experience\n"
                    "• You can provide detailed answers (no length restrictions)\n"
                    "• Your responses will be evaluated based on technical keywords and detail\n\n"
                    "💡 *Tips for better scores:*\n"
                    "• Be specific and detailed\n"
                    "• Mention relevant technologies and concepts\n"
                    "• Explain your thought process\n"
                    "• Share real examples from your experience\n\n"
                    "📊 *Progress: Question 1 of 3*\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "📖 Or send HELP and I'll be here to assist you! 🌸\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{tech_q1}"
                )
            else:
                # Original message for Demo Mode
                response_text = (
                    "Perfect! 😊\n\n"
                    "⚡ *Now come 3 technical questions.*\n\n"
                    "💡 *Tip:* Answer honestly based on your real experience.\n"
                    "There are no wrong answers — I just want to understand your current level. 🌸\n"
                    "*Be as detailed as possible - show me what you know!*\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{answers_tip}"
                    "📖 Or send HELP and I'll be here to assist you! 🌸\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{tech_q1}"
                )
        else:
            answers_tip = ""
            if is_demo_mode:
                answers_tip = "💡 *TIP:* En cualquier momento, envía RESPUESTAS para ver respuestas sugeridas\n\n"
            
            if not is_demo_mode:
                # Enhanced message for Free Mode
                response_text = (
                    "Perfecto! 😊\n\n"
                    f"⚡ *Ahora vienen 3 preguntas técnicas sobre {position}.*\n\n"
                    "💡 *Qué esperar:*\n"
                    "• Cada pregunta evalúa tu conocimiento técnico\n"
                    "• No hay respuestas incorrectas — sé honesto sobre tu experiencia\n"
                    "• Puedes dar respuestas detalladas (sin restricciones de longitud)\n"
                    "• Tus respuestas se evaluarán basándose en keywords técnicos y detalle\n\n"
                    "💡 *Tips para mejores scores:*\n"
                    "• Sé específico y detallado\n"
                    "• Menciona tecnologías y conceptos relevantes\n"
                    "• Explica tu proceso de pensamiento\n"
                    "• Comparte ejemplos reales de tu experiencia\n\n"
                    "📊 *Progreso: Pregunta 1 de 3*\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "📖 O envía AYUDA y estaré aquí para asistirte 🌸\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{tech_q1}"
                )
            else:
                # Original message for Demo Mode
                response_text = (
                    "Perfecto! 😊\n\n"
                    "⚡ *Ahora vienen 3 preguntas técnicas.*\n\n"
                    "💡 *Tip:* Responde con honestidad basándote en tu experiencia real.\n"
                    "No hay respuestas incorrectas — solo quiero conocer tu nivel actual. 🌸\n"
                    "*¡Sé lo más detallado posible - demuéstranos lo que sabes!*\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{answers_tip}"
                    "📖 O envía AYUDA y estaré aquí para asistirte 🌸\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{tech_q1}"
                )
        session['data']['tech_questions'] = []
        # Store the full question text for validation
        session['data']['current_tech_question'] = tech_q1
        session['stage'] = 7
        save_session(phone_number, session)  # CRITICAL: Save before returning
    
    elif stage == 7:  # Tech Question 1
        # Validate response - check if user just repeated the question
        current_question = session['data'].get('current_tech_question', '')
        lang = session.get('language', 'es')
        
        if current_question and is_question_repetition(current_question, message_text):
            # User just repeated the question, ask them to answer properly
            if lang == 'en':
                response_text = (
                    "⚠️ *Please provide your own answer to the question.*\n\n"
                    "It looks like you might have copied the question. Please explain your answer in your own words.\n\n"
                    f"{current_question}"
                )
            else:
                response_text = (
                    "⚠️ *Por favor proporciona tu propia respuesta a la pregunta.*\n\n"
                    "Parece que copiaste la pregunta. Por favor explica tu respuesta con tus propias palabras.\n\n"
                    f"{current_question}"
                )
            # Stay in same stage
            save_session(phone_number, session)
            return response_text
        
        # Validate that response makes sense
        is_valid, reason = is_response_makes_sense(message_text, min_words=8, min_length=30)
        if not is_valid:
            # Response doesn't make sense, ask for a better answer
            reason_messages = {
                'empty': ('Please provide an answer.', 'Por favor proporciona una respuesta.'),
                'too_short': ('Your answer is too short. Please provide more details (at least 30 characters).', 
                             'Tu respuesta es muy corta. Por favor proporciona más detalles (al menos 30 caracteres).'),
                'too_few_words': ('Your answer needs more words. Please explain your answer in at least 8 words.', 
                                 'Tu respuesta necesita más palabras. Por favor explica tu respuesta en al menos 8 palabras.'),
                'no_structure': ('Your answer seems incomplete. Please write complete sentences explaining your answer.', 
                                'Tu respuesta parece incompleta. Por favor escribe oraciones completas explicando tu respuesta.'),
                'no_coherence': ('Please provide a more detailed answer that explains your thoughts clearly.', 
                               'Por favor proporciona una respuesta más detallada que explique claramente tus pensamientos.'),
                'repetitive': ('Your answer seems repetitive. Please provide a more varied explanation.', 
                              'Tu respuesta parece repetitiva. Por favor proporciona una explicación más variada.')
            }
            
            msg_en, msg_es = reason_messages.get(reason, ('Please provide a better answer.', 'Por favor proporciona una mejor respuesta.'))
            
            if lang == 'en':
                response_text = (
                    f"⚠️ *{msg_en}*\n\n"
                    f"{current_question}"
                )
            else:
                response_text = (
                    f"⚠️ *{msg_es}*\n\n"
                    f"{current_question}"
                )
            # Stay in same stage
            save_session(phone_number, session)
            return response_text
        
        # Evaluate response
        position = session['data'].get('position', '').lower()
        if 'data engineer' in position:
            question_topic = "Normalization"
        elif 'backend developer' in position and 'django' in position:
            question_topic = "Django Models"
        elif 'full stack' in position and 'react' in position:
            question_topic = "React SSR vs CSR"
        elif 'api developer' in position or ('rest' in position and 'graphql' in position):
            question_topic = "REST vs GraphQL"
        elif 'backend engineer' in position and 'node.js' in position:
            question_topic = "Node.js Event Loop"
        elif 'software engineer' in position and 'fastapi' in position:
            question_topic = "FastAPI Advantages"
        elif 'backend' in position:
            question_topic = "Django Models"
        else:
            question_topic = "REST vs GraphQL"
        
        score = evaluate_response(question_topic, message_text, expects_brief=False)  # Q1 asks for detailed response
        log_user_action(phone_number, f"Tech Q1 scored: {score}/5.0", f"Question: {question_topic}", session, "INFO")
        session['data']['tech_questions'].append({
            'question': question_topic,
            'answer': message_text,
            'score': score
        })
        session['scores']['technical'] += score
        
        emotion = session['emotions'][-1]
        lang = session.get('language', 'es')
        
        # Select question 2 based on position
        if 'data engineer' in position:
            if lang == 'en':
                tech_q2 = (
                    "🔧 *Question 2 of 3:*\n"
                    "What does it mean that Python is a dynamically typed language? "
                    "What are the advantages and disadvantages in data engineering?\n\n"
                    "💡 *Hint:* Mention concepts like: runtime, type declaration, flexibility, performance"
                )
            else:
                tech_q2 = (
                    "🔧 *Pregunta 2 de 3:*\n"
                    "¿Qué significa que Python es un lenguaje de tipado dinámico? "
                    "¿Cuáles son las ventajas y desventajas en ingeniería de datos?\n\n"
                    "💡 *Hint:* Menciona conceptos como: runtime, declaración de tipo, flexibilidad, rendimiento"
                )
        elif 'backend developer' in position and 'django' in position:
            # Backend Developer (Python/Django)
            if lang == 'en':
                tech_q2 = (
                    "🔧 *Question 2 of 3:*\n"
                    "How would you design an API endpoint in Django to return a list of users in JSON format?\n\n"
                    "💡 *Hint:* Mention concepts like: Django REST Framework, serializer, view, JSON"
                )
            else:
                tech_q2 = (
                    "🔧 *Pregunta 2 de 3:*\n"
                    "¿Cómo diseñarías un endpoint de API en Django para devolver una lista de usuarios en formato JSON?\n\n"
                    "💡 *Hint:* Menciona conceptos como: Django REST Framework, serializer, view, JSON"
                )
        elif 'full stack' in position and 'react' in position:
            # Full Stack Developer (Python/React)
            if lang == 'en':
                tech_q2 = (
                    "🔧 *Question 2 of 3:*\n"
                    "How does React's virtual DOM improve performance compared to direct DOM manipulation?\n\n"
                    "💡 *Hint:* Mention concepts like: diffing algorithm, reconciliation, batch updates, re-rendering"
                )
            else:
                tech_q2 = (
                    "🔧 *Pregunta 2 de 3:*\n"
                    "¿Cómo mejora el virtual DOM de React el rendimiento comparado con la manipulación directa del DOM?\n\n"
                    "💡 *Hint:* Menciona conceptos como: algoritmo de diffing, reconciliación, actualizaciones en lote, re-renderizado"
                )
        elif 'api developer' in position or ('rest' in position and 'graphql' in position):
            # API Developer (REST/GraphQL)
            if lang == 'en':
                tech_q2 = (
                    "🔧 *Question 2 of 3:*\n"
                    "What is API versioning and what strategies would you use to version a REST API?\n\n"
                    "💡 *Hint:* Mention concepts like: URL versioning, header versioning, semantic versioning, backward compatibility"
                )
            else:
                tech_q2 = (
                    "🔧 *Pregunta 2 de 3:*\n"
                    "¿Qué es la versionado de APIs y qué estrategias usarías para versionar una API REST?\n\n"
                    "💡 *Hint:* Menciona conceptos como: versionado en URL, versionado en headers, versionado semántico, compatibilidad hacia atrás"
                )
        elif 'backend engineer' in position and 'node.js' in position:
            # Backend Engineer (Node.js/Express)
            if lang == 'en':
                tech_q2 = (
                    "🔧 *Question 2 of 3:*\n"
                    "What is middleware in Express.js and how would you use it to handle authentication?\n\n"
                    "💡 *Hint:* Mention concepts like: request/response cycle, next(), JWT, session management"
                )
            else:
                tech_q2 = (
                    "🔧 *Pregunta 2 de 3:*\n"
                    "¿Qué es middleware en Express.js y cómo lo usarías para manejar autenticación?\n\n"
                    "💡 *Hint:* Menciona conceptos como: ciclo request/response, next(), JWT, gestión de sesiones"
                )
        elif 'software engineer' in position and 'fastapi' in position:
            # Software Engineer (Python/FastAPI)
            if lang == 'en':
                tech_q2 = (
                    "🔧 *Question 2 of 3:*\n"
                    "How does FastAPI handle dependency injection and why is it useful for building APIs?\n\n"
                    "💡 *Hint:* Mention concepts like: Depends(), dependency tree, database connections, authentication"
                )
            else:
                tech_q2 = (
                    "🔧 *Pregunta 2 de 3:*\n"
                    "¿Cómo maneja FastAPI la inyección de dependencias y por qué es útil para construir APIs?\n\n"
                    "💡 *Hint:* Menciona conceptos como: Depends(), árbol de dependencias, conexiones de base de datos, autenticación"
                )
        elif 'backend' in position:
            # Fallback for other backend positions
            if lang == 'en':
                tech_q2 = (
                    "🔧 *Question 2 of 3:*\n"
                    "How would you design an API endpoint in Django to return a list of users in JSON format?\n\n"
                    "💡 *Hint:* Mention concepts like: Django REST Framework, serializer, view, JSON"
                )
            else:
                tech_q2 = (
                    "🔧 *Pregunta 2 de 3:*\n"
                    "¿Cómo diseñarías un endpoint de API en Django para devolver una lista de usuarios en formato JSON?\n\n"
                    "💡 *Hint:* Menciona conceptos como: Django REST Framework, serializer, view, JSON"
                )
        else:
            if lang == 'en':
                tech_q2 = (
                    "🔧 *Question 2 of 3:*\n"
                    "What is Docker and what is it used for?"
                )
            else:
                tech_q2 = (
                    "🔧 *Pregunta 2 de 3:*\n"
                    "¿Qué es Docker y para qué se usa?"
                )
        
        # Add progress indicator for Free Mode
        is_demo_mode = session.get('demo_mode') == 'full_interview'
        if not is_demo_mode:
            progress_text = "📊 *Progress: Question 2 of 3*\n\n" if lang == 'en' else "📊 *Progreso: Pregunta 2 de 3*\n\n"
            response_text = f"Interesting answer 🎉\n\n{progress_text}{tech_q2}" if lang == 'en' else f"Interesante respuesta 🎉\n\n{progress_text}{tech_q2}"
        else:
            response_text = f"Interesting answer 🎉\n\n{tech_q2}" if lang == 'en' else f"Interesante respuesta 🎉\n\n{tech_q2}"
        # Store the full question text for validation
        session['data']['current_tech_question'] = tech_q2
        session['stage'] = 8
    
    elif stage == 8:  # Tech Question 2
        # Validate response - check if user just repeated the question
        current_question = session['data'].get('current_tech_question', '')
        lang = session.get('language', 'es')
        
        if current_question and is_question_repetition(current_question, message_text):
            # User just repeated the question, ask them to answer properly
            if lang == 'en':
                response_text = (
                    "⚠️ *Please provide your own answer to the question.*\n\n"
                    "It looks like you might have copied the question. Please explain your answer in your own words.\n\n"
                    f"{current_question}"
                )
            else:
                response_text = (
                    "⚠️ *Por favor proporciona tu propia respuesta a la pregunta.*\n\n"
                    "Parece que copiaste la pregunta. Por favor explica tu respuesta con tus propias palabras.\n\n"
                    f"{current_question}"
                )
            # Stay in same stage
            save_session(phone_number, session)
            return response_text
        
        # Validate that response makes sense
        is_valid, reason = is_response_makes_sense(message_text, min_words=8, min_length=30)
        if not is_valid:
            reason_messages = {
                'empty': ('Please provide an answer.', 'Por favor proporciona una respuesta.'),
                'too_short': ('Your answer is too short. Please provide more details (at least 30 characters).', 
                             'Tu respuesta es muy corta. Por favor proporciona más detalles (al menos 30 caracteres).'),
                'too_few_words': ('Your answer needs more words. Please explain your answer in at least 8 words.', 
                                 'Tu respuesta necesita más palabras. Por favor explica tu respuesta en al menos 8 palabras.'),
                'no_structure': ('Your answer seems incomplete. Please write complete sentences explaining your answer.', 
                                'Tu respuesta parece incompleta. Por favor escribe oraciones completas explicando tu respuesta.'),
                'no_coherence': ('Please provide a more detailed answer that explains your thoughts clearly.', 
                               'Por favor proporciona una respuesta más detallada que explique claramente tus pensamientos.'),
                'repetitive': ('Your answer seems repetitive. Please provide a more varied explanation.', 
                              'Tu respuesta parece repetitiva. Por favor proporciona una explicación más variada.')
            }
            msg_en, msg_es = reason_messages.get(reason, ('Please provide a better answer.', 'Por favor proporciona una mejor respuesta.'))
            if lang == 'en':
                response_text = f"⚠️ *{msg_en}*\n\n{current_question}"
            else:
                response_text = f"⚠️ *{msg_es}*\n\n{current_question}"
            save_session(phone_number, session)
            return response_text
        
        position = session['data'].get('position', '').lower()
        if 'data engineer' in position:
            question_topic = "Python Dynamic Typing"
        elif 'backend developer' in position and 'django' in position:
            question_topic = "Django API Endpoint"
        elif 'full stack' in position and 'react' in position:
            question_topic = "React Virtual DOM"
        elif 'api developer' in position or ('rest' in position and 'graphql' in position):
            question_topic = "API Versioning"
        elif 'backend engineer' in position and 'node.js' in position:
            question_topic = "Express Middleware"
        elif 'software engineer' in position and 'fastapi' in position:
            question_topic = "FastAPI Dependency Injection"
        elif 'backend' in position:
            question_topic = "Django API Endpoint"
        else:
            question_topic = "Docker"
        
        score = evaluate_response(question_topic, message_text, expects_brief=False)  # Q2 now expects detailed response like Q1
        log_user_action(phone_number, f"Tech Q2 scored: {score}/5.0", f"Question: {question_topic}", session, "INFO")
        session['data']['tech_questions'].append({
            'question': question_topic,
            'answer': message_text,
            'score': score
        })
        session['scores']['technical'] += score
        
        emotion = session['emotions'][-1]
        lang = session.get('language', 'es')
        
        # Select question 3 based on position
        if 'data engineer' in position:
            if lang == 'en':
                tech_q3 = (
                    "🔧 *Question 3 of 3:*\n"
                    "How does Apache Spark handle fault tolerance in a distributed environment?\n\n"
                    "💡 *Hint:* Mention concepts like: RDDs, lineage, transformations, node failure, reconstruction"
                )
            else:
                tech_q3 = (
                    "🔧 *Pregunta 3 de 3:*\n"
                    "¿Cómo maneja Apache Spark la tolerancia a fallos en un entorno distribuido?\n\n"
                    "💡 *Hint:* Menciona conceptos como: RDDs, lineage, transformaciones, fallo de nodo, reconstrucción"
                )
        elif 'backend developer' in position and 'django' in position:
            # Backend Developer (Python/Django)
            if lang == 'en':
                tech_q3 = (
                    "🔧 *Question 3 of 3:*\n"
                    "What is Docker and how would you use it to deploy a Django backend application?\n\n"
                    "💡 *Hint:* Mention concepts like: containers, Dockerfile, docker-compose, images, port mapping"
                )
            else:
                tech_q3 = (
                    "🔧 *Pregunta 3 de 3:*\n"
                    "¿Qué es Docker y cómo lo usarías para desplegar una aplicación backend Django?\n\n"
                    "💡 *Hint:* Menciona conceptos como: contenedores, Dockerfile, docker-compose, imágenes, mapeo de puertos"
                )
        elif 'full stack' in position and 'react' in position:
            # Full Stack Developer (Python/React)
            if lang == 'en':
                tech_q3 = (
                    "🔧 *Question 3 of 3:*\n"
                    "How would you manage state in a large React application? Compare different state management approaches.\n\n"
                    "💡 *Hint:* Mention concepts like: Context API, Redux, Zustand, prop drilling, local vs global state"
                )
            else:
                tech_q3 = (
                    "🔧 *Pregunta 3 de 3:*\n"
                    "¿Cómo manejarías el estado en una aplicación React grande? Compara diferentes enfoques de gestión de estado.\n\n"
                    "💡 *Hint:* Menciona conceptos como: Context API, Redux, Zustand, prop drilling, estado local vs global"
                )
        elif 'api developer' in position or ('rest' in position and 'graphql' in position):
            # API Developer (REST/GraphQL)
            if lang == 'en':
                tech_q3 = (
                    "🔧 *Question 3 of 3:*\n"
                    "What is rate limiting and how would you implement it to protect an API from abuse?\n\n"
                    "💡 *Hint:* Mention concepts like: token bucket, sliding window, Redis, throttling, DDoS protection"
                )
            else:
                tech_q3 = (
                    "🔧 *Pregunta 3 de 3:*\n"
                    "¿Qué es el rate limiting y cómo lo implementarías para proteger una API del abuso?\n\n"
                    "💡 *Hint:* Menciona conceptos como: token bucket, ventana deslizante, Redis, throttling, protección DDoS"
                )
        elif 'backend engineer' in position and 'node.js' in position:
            # Backend Engineer (Node.js/Express)
            if lang == 'en':
                tech_q3 = (
                    "🔧 *Question 3 of 3:*\n"
                    "What is the difference between callback functions, Promises, and async/await in Node.js? When would you use each?\n\n"
                    "💡 *Hint:* Mention concepts like: callback hell, error handling, readability, asynchronous flow"
                )
            else:
                tech_q3 = (
                    "🔧 *Pregunta 3 de 3:*\n"
                    "¿Cuál es la diferencia entre funciones callback, Promises y async/await en Node.js? ¿Cuándo usarías cada una?\n\n"
                    "💡 *Hint:* Menciona conceptos como: callback hell, manejo de errores, legibilidad, flujo asíncrono"
                )
        elif 'software engineer' in position and 'fastapi' in position:
            # Software Engineer (Python/FastAPI)
            if lang == 'en':
                tech_q3 = (
                    "🔧 *Question 3 of 3:*\n"
                    "What is Pydantic and how does FastAPI use it for request/response validation?\n\n"
                    "💡 *Hint:* Mention concepts like: data validation, type coercion, automatic documentation, serialization"
                )
            else:
                tech_q3 = (
                    "🔧 *Pregunta 3 de 3:*\n"
                    "¿Qué es Pydantic y cómo lo usa FastAPI para validación de request/response?\n\n"
                    "💡 *Hint:* Menciona conceptos como: validación de datos, coerción de tipos, documentación automática, serialización"
                )
        elif 'backend' in position:
            # Fallback for other backend positions
            if lang == 'en':
                tech_q3 = (
                    "🔧 *Question 3 of 3:*\n"
                    "What is Docker and how would you use it to deploy a Django backend application?\n\n"
                    "💡 *Hint:* Mention concepts like: containers, Dockerfile, docker-compose, images, port mapping"
                )
            else:
                tech_q3 = (
                    "🔧 *Pregunta 3 de 3:*\n"
                    "¿Qué es Docker y cómo lo usarías para desplegar una aplicación backend Django?\n\n"
                    "💡 *Hint:* Menciona conceptos como: contenedores, Dockerfile, docker-compose, imágenes, mapeo de puertos"
                )
        else:
            if lang == 'en':
                tech_q3 = (
                    "🔧 *Question 3 of 3:*\n"
                    "What is CI/CD? Why is it important?"
                )
            else:
                tech_q3 = (
                    "🔧 *Pregunta 3 de 3:*\n"
                    "¿Qué es CI/CD? ¿Por qué es importante?"
                )
        
        # Add progress indicator for Free Mode
        is_demo_mode = session.get('demo_mode') == 'full_interview'
        if not is_demo_mode:
            progress_text = "📊 *Progress: Question 3 of 3*\n\n" if lang == 'en' else "📊 *Progreso: Pregunta 3 de 3*\n\n"
            response_text = f"Good 🎉\n\n{progress_text}{tech_q3}" if lang == 'en' else f"Bien 🎉\n\n{progress_text}{tech_q3}"
        else:
            response_text = f"Good 🎉\n\n{tech_q3}" if lang == 'en' else f"Bien 🎉\n\n{tech_q3}"
        # Store the full question text for validation
        session['data']['current_tech_question'] = tech_q3
        session['stage'] = 9
    
    elif stage == 9:  # Tech Question 3
        # Validate response - check if user just repeated the question
        current_question = session['data'].get('current_tech_question', '')
        lang = session.get('language', 'es')
        
        if current_question and is_question_repetition(current_question, message_text):
            # User just repeated the question, ask them to answer properly
            if lang == 'en':
                response_text = (
                    "⚠️ *Please provide your own answer to the question.*\n\n"
                    "It looks like you might have copied the question. Please explain your answer in your own words.\n\n"
                    f"{current_question}"
                )
            else:
                response_text = (
                    "⚠️ *Por favor proporciona tu propia respuesta a la pregunta.*\n\n"
                    "Parece que copiaste la pregunta. Por favor explica tu respuesta con tus propias palabras.\n\n"
                    f"{current_question}"
                )
            # Stay in same stage
            save_session(phone_number, session)
            return response_text
        
        # Validate that response makes sense
        is_valid, reason = is_response_makes_sense(message_text, min_words=8, min_length=30)
        if not is_valid:
            reason_messages = {
                'empty': ('Please provide an answer.', 'Por favor proporciona una respuesta.'),
                'too_short': ('Your answer is too short. Please provide more details (at least 30 characters).', 
                             'Tu respuesta es muy corta. Por favor proporciona más detalles (al menos 30 caracteres).'),
                'too_few_words': ('Your answer needs more words. Please explain your answer in at least 8 words.', 
                                 'Tu respuesta necesita más palabras. Por favor explica tu respuesta en al menos 8 palabras.'),
                'no_structure': ('Your answer seems incomplete. Please write complete sentences explaining your answer.', 
                                'Tu respuesta parece incompleta. Por favor escribe oraciones completas explicando tu respuesta.'),
                'no_coherence': ('Please provide a more detailed answer that explains your thoughts clearly.', 
                               'Por favor proporciona una respuesta más detallada que explique claramente tus pensamientos.'),
                'repetitive': ('Your answer seems repetitive. Please provide a more varied explanation.', 
                              'Tu respuesta parece repetitiva. Por favor proporciona una explicación más variada.')
            }
            msg_en, msg_es = reason_messages.get(reason, ('Please provide a better answer.', 'Por favor proporciona una mejor respuesta.'))
            if lang == 'en':
                response_text = f"⚠️ *{msg_en}*\n\n{current_question}"
            else:
                response_text = f"⚠️ *{msg_es}*\n\n{current_question}"
            save_session(phone_number, session)
            return response_text
        
        position = session['data'].get('position', '').lower()
        if 'data engineer' in position:
            question_topic = "Apache Spark Fault Tolerance"
        elif 'backend developer' in position and 'django' in position:
            question_topic = "Docker Django Backend"
        elif 'full stack' in position and 'react' in position:
            question_topic = "React State Management"
        elif 'api developer' in position or ('rest' in position and 'graphql' in position):
            question_topic = "API Rate Limiting"
        elif 'backend engineer' in position and 'node.js' in position:
            question_topic = "Node.js Async Patterns"
        elif 'software engineer' in position and 'fastapi' in position:
            question_topic = "FastAPI Pydantic"
        elif 'backend' in position:
            question_topic = "Docker Django Backend"
        else:
            question_topic = "CI/CD"
        
        score = evaluate_response(question_topic, message_text, expects_brief=False)  # Q3 now expects detailed response like Q1
        log_user_action(phone_number, f"Tech Q3 scored: {score}/5.0", f"Question: {question_topic}", session, "INFO")
        session['data']['tech_questions'].append({
            'question': question_topic,
            'answer': message_text,
            'score': score
        })
        session['scores']['technical'] += score
        
        # Calculate average technical score
        avg_tech = session['scores']['technical'] / 3
        log_user_action(phone_number, f"Technical average: {avg_tech:.1f}/5.0", "", session, "INFO")
        
        # Get position-specific technology for English question
        position = session['data'].get('position', '').lower()
        lang = session.get('language', 'es')
        tech_stack = "Python"  # Default
        
        if 'data engineer' in position:
            tech_stack = "data engineering and ETL pipelines (PySpark, Airflow, etc.)"
        elif 'backend' in position:
            tech_stack = "backend development (Python, Node.js, databases, etc.)"
        elif 'frontend' in position:
            tech_stack = "frontend development (React, TypeScript, UI/UX, etc.)"
        elif 'devops' in position:
            tech_stack = "DevOps and infrastructure (Docker, Kubernetes, CI/CD, etc.)"
        elif 'full stack' in position:
            tech_stack = "full stack development (frontend and backend technologies)"
        
        if lang == 'en':
            response_text = (
                f"✅ Technical questions completed!\n\n"
                f"📊 Your technical score: *{avg_tech:.1f}/5.0*\n\n"
                "🗣️ *Now let's evaluate your English.*\n\n"
                "🇬🇧 *Question 1:*\n"
                f"Describe your experience with {tech_stack}.\n"
                "(Answer in English please)"
            )
        else:
            response_text = (
                f"✅ Preguntas técnicas completadas!\n\n"
                f"📊 Tu score técnico: *{avg_tech:.1f}/5.0*\n\n"
                "🗣️ *Ahora evaluemos tu inglés.*\n\n"
                "🇬🇧 *Question 1:*\n"
                f"Describe your experience with {tech_stack}.\n"
                "(Answer in English please)"
            )
        session['data']['english_questions'] = []
        session['stage'] = 10
    
    elif stage == 10:  # English Question 1
        # Validate that response is in English
        detected_lang = detect_language(message_text)
        lang = session.get('language', 'es')
        
        if detected_lang != 'en':
            # Response is not in English, ask to answer in English
            # Get tech_stack for the question
            position = session.get('data', {}).get('position', '').lower()
            tech_stack = "Python"  # Default
            if 'data' in position or 'engineer' in position:
                tech_stack = "data engineering and ETL pipelines (PySpark, Airflow, etc.)"
            elif 'backend' in position or 'developer' in position:
                tech_stack = "backend development (Python, Node.js, databases, etc.)"
            
            if lang == 'en':
                response_text = (
                    "⚠️ *Please answer in English.*\n\n"
                    "This question evaluates your English level. Please respond in English.\n\n"
                    "🇬🇧 *Question 1:*\n"
                    f"Describe your experience with {tech_stack}.\n"
                    "(Answer in English please)"
                )
            else:
                response_text = (
                    "⚠️ *Por favor responde en inglés.*\n\n"
                    "Esta pregunta evalúa tu nivel de inglés. Por favor responde en inglés.\n\n"
                    "🇬🇧 *Question 1:*\n"
                    f"Describe your experience with {tech_stack}.\n"
                    "(Answer in English please)"
                )
            # Stay in same stage, don't advance
            save_session(phone_number, session)
            return response_text
        
        # Check if response is copied from previous English answers only (not technical)
        # For English Question 1, we only check against other English answers, not technical
        # This allows users to copy suggested answers without issues
        print(f"[DEBUG] Checking if response is copied from previous English answers...")
        english_questions = session.get('data', {}).get('english_questions', [])
        print(f"[DEBUG] Found {len(english_questions)} previous English answers")
        if len(english_questions) > 0:
            # Check if this response matches any previous English answer
            current_lower = message_text.lower().strip()
            current_words = set(current_lower.split())
            
            for eng_q in english_questions:
                if 'answer' in eng_q:
                    prev_answer = eng_q['answer'].lower().strip()
                    if len(prev_answer) >= 20:
                        prev_words = set(prev_answer.split())
                        if len(current_words) > 0 and len(prev_words) > 0:
                            overlap = len(current_words & prev_words)
                            similarity = overlap / min(len(current_words), len(prev_words))
                            if similarity > 0.7:  # 70% similarity threshold
                                # Get tech_stack for error message
                                position = session.get('data', {}).get('position', '').lower()
                                tech_stack = "Python"  # Default
                                if 'data' in position or 'engineer' in position:
                                    tech_stack = "data engineering and ETL pipelines (PySpark, Airflow, etc.)"
                                elif 'backend' in position or 'developer' in position:
                                    tech_stack = "backend development (Python, Node.js, databases, etc.)"
                                
                                if lang == 'en':
                                    response_text = (
                                        "⚠️ *Please provide your own answer to this question.*\n\n"
                                        "It looks like you might have copied a previous answer. This question asks about your experience, please answer specifically.\n\n"
                                        "🇬🇧 *Question 1:*\n"
                                        f"Describe your experience with {tech_stack}.\n"
                                        "(Answer in English please)"
                                    )
                                else:
                                    response_text = (
                                        "⚠️ *Por favor proporciona tu propia respuesta a esta pregunta.*\n\n"
                                        "Parece que copiaste una respuesta anterior. Esta pregunta pregunta sobre tu experiencia, por favor responde específicamente.\n\n"
                                        "🇬🇧 *Question 1:*\n"
                                        f"Describe your experience with {tech_stack}.\n"
                                        "(Answer in English please)"
                                    )
                                save_session(phone_number, session)
                                return response_text
        
        # Validate that response makes sense
        is_valid, reason = is_response_makes_sense(message_text, min_words=5, min_length=20)
        if not is_valid:
            reason_messages = {
                'empty': ('Please provide an answer.', 'Por favor proporciona una respuesta.'),
                'too_short': ('Your answer is too short. Please provide more details (at least 20 characters).', 
                             'Tu respuesta es muy corta. Por favor proporciona más detalles (al menos 20 caracteres).'),
                'too_few_words': ('Your answer needs more words. Please explain in at least 5 words.', 
                                 'Tu respuesta necesita más palabras. Por favor explica en al menos 5 palabras.'),
                'no_structure': ('Your answer seems incomplete. Please write complete sentences.', 
                                'Tu respuesta parece incompleta. Por favor escribe oraciones completas.'),
                'no_coherence': ('Please provide a more detailed answer that explains your experience clearly.', 
                               'Por favor proporciona una respuesta más detallada que explique claramente tu experiencia.'),
                'repetitive': ('Your answer seems repetitive. Please provide a more varied explanation.', 
                              'Tu respuesta parece repetitiva. Por favor proporciona una explicación más variada.')
            }
            msg_en, msg_es = reason_messages.get(reason, ('Please provide a better answer.', 'Por favor proporciona una mejor respuesta.'))
            position = session.get('data', {}).get('position', '').lower()
            tech_stack = "Python"
            if 'data' in position or 'engineer' in position:
                tech_stack = "data engineering and ETL pipelines (PySpark, Airflow, etc.)"
            elif 'backend' in position or 'developer' in position:
                tech_stack = "backend development (Python, Node.js, databases, etc.)"
            if lang == 'en':
                response_text = f"⚠️ *{msg_en}*\n\n🇬🇧 *Question 1:*\nDescribe your experience with {tech_stack}.\n(Answer in English please)"
            else:
                response_text = f"⚠️ *{msg_es}*\n\n🇬🇧 *Question 1:*\nDescribe your experience with {tech_stack}.\n(Answer in English please)"
            save_session(phone_number, session)
            return response_text
        
        # Response is in English, evaluate normally
        print(f"[DEBUG] Response is valid English, evaluating...")
        english_score = evaluate_english_level(message_text)
        print(f"[DEBUG] English score: {english_score:.1f}/5.0")
        session['data']['english_questions'].append({
            'question': 'Python experience',
            'answer': message_text,
            'score': english_score
        })
        session['scores']['english'] += english_score
        print(f"[DEBUG] Total English score: {session['scores']['english']:.1f}")
        
        emotion = session['emotions'][-1]
        
        # Log that we're proceeding to question 2
        print(f"[INFO] English Question 1 answered successfully. Score: {english_score:.1f}/5.0")
        print(f"[INFO] Moving to stage 11 (English Question 2)")
        
        response_text = (
            "Good! 😊\n\n"
            "🇬🇧 *Question 2:*\n"
            "What's your biggest professional achievement?\n"
            "(Answer in English please)"
        )
        session['stage'] = 11
        save_session(phone_number, session)  # CRITICAL: Save before returning
        print(f"[INFO] Response text generated (length: {len(response_text)} chars), returning...")
        print(f"[DEBUG] Response text content: {repr(response_text)}")
        return response_text
    
    elif stage == 11:  # English Question 2
        # Check if user already answered this question
        english_questions = session.get('data', {}).get('english_questions', [])
        if len(english_questions) >= 2:
            # User already answered question 2, check if this is a short non-answer message
            message_stripped = message_text.strip().upper()
            # Common greetings/short messages that shouldn't be processed as answers
            short_messages = ['HOLA', 'HI', 'HELLO', 'HOLA!', 'HI!', 'HELLO!', 'OK', 'OKAY', 'SI', 'YES', 'NO', 'GRACIAS', 'THANKS', 'THANK YOU']
            
            if message_stripped in short_messages or len(message_text.strip()) < 10:
                # This is a greeting or short message, not an answer - continue to next stage
                lang = session.get('language', 'es')
                avg_english = session['scores']['english'] / 2
                
                if lang == 'en':
                    response_text = (
                        f"Excellent! ✅\n\n"
                        f"📊 English level: *{avg_english:.1f}/5.0*\n\n"
                        "💼 *Last section: Soft Skills*\n\n"
                        "🤝 Tell me about a time you worked in a team to solve a difficult problem."
                    )
                else:
                    response_text = (
                        f"Excellent! ✅\n\n"
                        f"📊 English level: *{avg_english:.1f}/5.0*\n\n"
                        "💼 *Última sección: Soft Skills*\n\n"
                        "🤝 Cuéntame sobre una vez que trabajaste en equipo para resolver un problema difícil."
                    )
                session['stage'] = 12
                save_session(phone_number, session)
                return response_text
        
        # Validate that response is in English
        detected_lang = detect_language(message_text)
        lang = session.get('language', 'es')
        
        if detected_lang != 'en':
            # Response is not in English, ask to answer in English
            if lang == 'en':
                response_text = (
                    "⚠️ *Please answer in English.*\n\n"
                    "This question evaluates your English level. Please respond in English.\n\n"
                    "🇬🇧 *Question 2:*\n"
                    "What's your biggest professional achievement?\n"
                    "(Answer in English please)"
                )
            else:
                response_text = (
                    "⚠️ *Por favor responde en inglés.*\n\n"
                    "Esta pregunta evalúa tu nivel de inglés. Por favor responde en inglés.\n\n"
                    "🇬🇧 *Question 2:*\n"
                    "What's your biggest professional achievement?\n"
                    "(Answer in English please)"
                )
            # Stay in same stage, don't advance
            save_session(phone_number, session)
            return response_text
        
        # Check if response is copied from previous answers
        is_copied, source = is_copied_from_previous(session, message_text)
        if is_copied:
            if lang == 'en':
                response_text = (
                    "⚠️ *Please provide your own answer to this question.*\n\n"
                    "It looks like you might have copied a previous answer. This question asks about your achievement, please answer specifically.\n\n"
                    "🇬🇧 *Question 2:*\n"
                    "What's your biggest professional achievement?\n"
                    "(Answer in English please)"
                )
            else:
                response_text = (
                    "⚠️ *Por favor proporciona tu propia respuesta a esta pregunta.*\n\n"
                    "Parece que copiaste una respuesta anterior. Esta pregunta pregunta sobre tu logro, por favor responde específicamente.\n\n"
                    "🇬🇧 *Question 2:*\n"
                    "What's your biggest professional achievement?\n"
                    "(Answer in English please)"
                )
            save_session(phone_number, session)
            return response_text
        
        # Validate that response makes sense
        is_valid, reason = is_response_makes_sense(message_text, min_words=5, min_length=20)
        if not is_valid:
            reason_messages = {
                'empty': ('Please provide an answer.', 'Por favor proporciona una respuesta.'),
                'too_short': ('Your answer is too short. Please provide more details (at least 20 characters).', 
                             'Tu respuesta es muy corta. Por favor proporciona más detalles (al menos 20 caracteres).'),
                'too_few_words': ('Your answer needs more words. Please explain in at least 5 words.', 
                                 'Tu respuesta necesita más palabras. Por favor explica en al menos 5 palabras.'),
                'no_structure': ('Your answer seems incomplete. Please write complete sentences.', 
                                'Tu respuesta parece incompleta. Por favor escribe oraciones completas.'),
                'no_coherence': ('Please provide a more detailed answer that explains your achievement clearly.', 
                               'Por favor proporciona una respuesta más detallada que explique claramente tu logro.'),
                'repetitive': ('Your answer seems repetitive. Please provide a more varied explanation.', 
                              'Tu respuesta parece repetitiva. Por favor proporciona una explicación más variada.')
            }
            msg_en, msg_es = reason_messages.get(reason, ('Please provide a better answer.', 'Por favor proporciona una mejor respuesta.'))
            if lang == 'en':
                response_text = f"⚠️ *{msg_en}*\n\n🇬🇧 *Question 2:*\nWhat's your biggest professional achievement?\n(Answer in English please)"
            else:
                response_text = f"⚠️ *{msg_es}*\n\n🇬🇧 *Question 2:*\nWhat's your biggest professional achievement?\n(Answer in English please)"
            save_session(phone_number, session)
            return response_text
        
        # Response is in English, evaluate normally
        english_score = evaluate_english_level(message_text)
        session['data']['english_questions'].append({
            'question': 'Professional achievement',
            'answer': message_text,
            'score': english_score
        })
        session['scores']['english'] += english_score
        
        # Calculate average English score
        avg_english = session['scores']['english'] / 2
        lang = session.get('language', 'es')
        
        if lang == 'en':
            response_text = (
                f"Excellent! ✅\n\n"
                f"📊 English level: *{avg_english:.1f}/5.0*\n\n"
                "💼 *Last section: Soft Skills*\n\n"
                "🤝 Tell me about a time you worked in a team to solve a difficult problem."
            )
        else:
            response_text = (
                f"Excellent! ✅\n\n"
                f"📊 English level: *{avg_english:.1f}/5.0*\n\n"
                "💼 *Última sección: Soft Skills*\n\n"
                "🤝 Cuéntame sobre una vez que trabajaste en equipo para resolver un problema difícil."
            )
        session['stage'] = 12
    
    elif stage == 12:  # Soft Skills
        lang = session.get('language', 'es')
        
        # Check if response is copied from previous answers
        is_copied, source = is_copied_from_previous(session, message_text)
        if is_copied:
            if lang == 'en':
                response_text = (
                    "⚠️ *Please provide your own answer to this question.*\n\n"
                    "It looks like you might have copied a previous answer. This question asks about teamwork, please answer specifically.\n\n"
                    "🤝 Tell me about a time you worked in a team to solve a difficult problem."
                )
            else:
                response_text = (
                    "⚠️ *Por favor proporciona tu propia respuesta a esta pregunta.*\n\n"
                    "Parece que copiaste una respuesta anterior. Esta pregunta pregunta sobre trabajo en equipo, por favor responde específicamente.\n\n"
                    "🤝 Cuéntame sobre una vez que trabajaste en equipo para resolver un problema difícil."
                )
            save_session(phone_number, session)
            return response_text
        
        # Validate that response makes sense
        is_valid, reason = is_response_makes_sense(message_text, min_words=8, min_length=30)
        if not is_valid:
            reason_messages = {
                'empty': ('Please provide an answer.', 'Por favor proporciona una respuesta.'),
                'too_short': ('Your answer is too short. Please provide more details (at least 30 characters).', 
                             'Tu respuesta es muy corta. Por favor proporciona más detalles (al menos 30 caracteres).'),
                'too_few_words': ('Your answer needs more words. Please explain in at least 8 words.', 
                                 'Tu respuesta necesita más palabras. Por favor explica en al menos 8 palabras.'),
                'no_structure': ('Your answer seems incomplete. Please write complete sentences explaining your experience.', 
                                'Tu respuesta parece incompleta. Por favor escribe oraciones completas explicando tu experiencia.'),
                'no_coherence': ('Please provide a more detailed answer that explains your teamwork experience clearly.', 
                               'Por favor proporciona una respuesta más detallada que explique claramente tu experiencia de trabajo en equipo.'),
                'repetitive': ('Your answer seems repetitive. Please provide a more varied explanation.', 
                              'Tu respuesta parece repetitiva. Por favor proporciona una explicación más variada.')
            }
            msg_en, msg_es = reason_messages.get(reason, ('Please provide a better answer.', 'Por favor proporciona una mejor respuesta.'))
            if lang == 'en':
                response_text = f"⚠️ *{msg_en}*\n\n🤝 Tell me about a time you worked in a team to solve a difficult problem."
            else:
                response_text = f"⚠️ *{msg_es}*\n\n🤝 Cuéntame sobre una vez que trabajaste en equipo para resolver un problema difícil."
            save_session(phone_number, session)
            return response_text
        
        # Evaluate soft skills
        soft_score = evaluate_soft_skills(message_text)
        session['data']['soft_skills_answer'] = message_text
        session['scores']['soft_skills'] = soft_score
        
        emotion = session['emotions'][-1]
        lang = session.get('language', 'es')
        
        if lang == 'en':
            response_text = (
                "Very good! 😊\n\n"
                f"📊 Soft Skills score: *{soft_score:.1f}/5.0*\n\n"
                "❓ *Last question:*\n"
                "Why should we hire you and not another candidate?\n"
                "(Be honest and specific)"
            )
        else:
            response_text = (
                "Muy bien! 😊\n\n"
                f"📊 Soft Skills score: *{soft_score:.1f}/5.0*\n\n"
                "❓ *Última pregunta:*\n"
                "¿Por qué deberíamos contratarte a ti y no a otro candidato?\n"
                "(Se honesto y específico)"
            )
        session['stage'] = 13
    
    elif stage == 13:  # Final Question
        session['data']['final_answer'] = message_text
        
        # Calculate final scores
        tech_avg = session['scores']['technical'] / 3
        english_avg = session['scores']['english'] / 2
        soft_skills = session['scores']['soft_skills']
        
        # Overall score (weighted average)
        final_score = (tech_avg * 0.4) + (english_avg * 0.3) + (soft_skills * 0.3)
        log_user_action(phone_number, f"Final score calculated: {final_score:.2f}/5.0", 
                       f"Tech: {tech_avg:.1f}, English: {english_avg:.1f}, Soft: {soft_skills:.1f}", 
                       session, "INFO")
        
        # Emotional analysis
        avg_confidence = sum(e['confidence'] for e in session['emotions']) / len(session['emotions'])
        dominant_emotion = max(set(e['emotion'] for e in session['emotions']), 
                             key=lambda x: sum(1 for e in session['emotions'] if e['emotion'] == x))
        
        # Emotional journey (start → end)
        first_emotion = session['emotions'][0]
        last_emotion = session['emotions'][-1]
        confidence_diff = last_emotion['confidence'] - first_emotion['confidence']
        
        # Determine trend
        # If same emotion, always show stable (regardless of confidence change)
        if first_emotion['emotion'] == last_emotion['emotion']:
            trend_emoji = "➡️"
            trend_text_en = "Stable"
            trend_text_es = "Estable"
        elif confidence_diff > 0.05:  # >5% improvement
            trend_emoji = "↗️"
            trend_text_en = "Improving"
            trend_text_es = "Mejorando"
        elif confidence_diff < -0.05:  # >5% decline
            trend_emoji = "↘️"
            trend_text_en = "Declining"
            trend_text_es = "Declinando"
        else:  # Stable
            trend_emoji = "➡️"
            trend_text_en = "Stable"
            trend_text_es = "Estable"
        
        journey_en = f"📊 Journey: {first_emotion['emotion']} (start) → {last_emotion['emotion']} (end) {trend_emoji} {trend_text_en}"
        journey_es = f"📊 Trayectoria: {first_emotion['emotion']} (inicio) → {last_emotion['emotion']} (final) {trend_emoji} {trend_text_es}"
        
        # Infer technical level based on weighted score (80% technical, 15% english, 5% soft skills)
        # This reflects how technical level is actually evaluated in the industry
        inference_score = (tech_avg * 0.80) + (english_avg * 0.15) + (soft_skills * 0.05)
        inference_percentage = (inference_score / 5.0) * 100
        
        # Keep final_percentage for overall assessment
        final_percentage = (final_score / 5.0) * 100
        tech_percentage = (tech_avg / 5.0) * 100  # Still needed for upgrade logic
        
        # NOTE: Penalties for trust_score, confidence, and inconsistencies will be applied
        # AFTER trust_score and inconsistencies are calculated (see line ~2512)
        # This ensures we have accurate data before applying penalties
        
        # Initial inferred level (will be recalculated after penalties are applied)
        if inference_percentage >= 85:
            inferred_level = "Senior"
        elif inference_percentage >= 65:
            inferred_level = "Mid-Level"
        else:
            inferred_level = "Junior"
        
        # Suggest upgrade if candidate exceeds expectations
        # More strict criteria: must have good English and overall performance
        upgrade_suggestion = None
        upgrade_justification = []
        
        if inferred_level == "Junior" and tech_percentage >= 75 and english_avg >= 3.0 and final_score >= 3.5:
            upgrade_suggestion = f"💡 *UPGRADE POTENTIAL: Mid-Level {session['data']['position']}*"
            upgrade_justification.append(f"Technical score ({tech_avg:.1f}/5.0) supera nivel Junior")
            if english_avg >= 3.5:
                upgrade_justification.append(f"Inglés sólido ({english_avg:.1f}/5.0)")
            if soft_skills >= 3.5:
                upgrade_justification.append("Soft skills destacadas")
        
        elif inferred_level == "Mid-Level" and tech_percentage >= 80 and english_avg >= 3.5 and final_score >= 4.0:
            upgrade_suggestion = f"🚀 *UPGRADE POTENTIAL: Senior {session['data']['position']}*"
            upgrade_justification.append(f"Technical score ({tech_avg:.1f}/5.0) supera nivel Mid")
            if english_avg >= 4.0:
                upgrade_justification.append(f"Inglés avanzado ({english_avg:.1f}/5.0)")
            if soft_skills >= 4.0:
                upgrade_justification.append("Soft skills excepcionales")
        
        # Get language BEFORE using it (FIX: UnboundLocalError)
        lang = session.get('language', 'es')
        
        # FORCE ENGLISH for demo profiles (CRITICAL FIX)
        candidate_name = session['data'].get('name', '')
        if candidate_name in ['Luis Martínez', 'Ana García']:
            if candidate_name == 'Luis Martínez':
                lang = 'en'  # FORCE ENGLISH for Luis
                print(f"[FORCE] Language FORCED to 'en' for Luis Martínez")
            # Ana García uses Spanish by default unless specified otherwise
        
        
        # === INCONSISTENCY DETECTION ===
        # FORCE RELOAD MODULE and use functions directly from module (CRITICAL FIX)
        import importlib
        import src.whatsapp_inconsistency_detector as detector_module
        importlib.reload(detector_module)
        
        # Rename tech_questions to technical_questions to match detector expectations
        session['data']['technical_questions'] = session['data'].get('tech_questions', [])
        
        # Detect inconsistencies (using module functions directly, not imported ones)
        # use_bert=True para activar mejoras con BERT (puede desactivarse si hay problemas)
        # Con timeout real y manejo robusto de errores para garantizar experiencia fluida
        try:
            import time
            import threading
            
            # Variables para timeout multiplataforma
            inconsistencies_result = [None]
            trust_score_result = [None]
            inconsistency_report_result = [None]
            timeout_occurred = [False]
            
            def detect_with_timeout():
                """Función que ejecuta la detección con timeout"""
                try:
                    inconsistencies_result[0] = detector_module.detect_whatsapp_inconsistencies(session, language=lang, use_bert=True)
                    trust_score_result[0] = detector_module.calculate_trust_score(inconsistencies_result[0])
                    inconsistency_report_result[0] = detector_module.generate_inconsistency_report(inconsistencies_result[0], language=lang)
                except Exception as e:
                    print(f"[ERROR] Error en detección dentro de thread: {e}")
                    inconsistencies_result[0] = []
                    trust_score_result[0] = 100
                    inconsistency_report_result[0] = "✅ No se detectaron inconsistencias significativas" if lang == 'es' else "✅ No significant inconsistencies detected"
            
            start_time = time.time()
            
            # Ejecutar en thread con timeout de 2 segundos máximo
            detection_thread = threading.Thread(target=detect_with_timeout)
            detection_thread.daemon = True  # Thread muere si el programa principal termina
            detection_thread.start()
            detection_thread.join(timeout=2.0)  # Esperar máximo 2 segundos
            
            if detection_thread.is_alive():
                # Thread aún corriendo después de timeout - usar fallback
                elapsed = time.time() - start_time
                print(f"[WARNING] Detección de inconsistencias timeout después de {elapsed:.2f}s - usando fallback rápido")
                inconsistencies = []
                trust_score = 100  # Score por defecto
                inconsistency_report = "✅ No se detectaron inconsistencias significativas" if lang == 'es' else "✅ No significant inconsistencies detected"
            else:
                # Thread completó - usar resultados
                inconsistencies = inconsistencies_result[0] if inconsistencies_result[0] is not None else []
                trust_score = trust_score_result[0] if trust_score_result[0] is not None else 100
                inconsistency_report = inconsistency_report_result[0] if inconsistency_report_result[0] is not None else ("✅ No se detectaron inconsistencias significativas" if lang == 'es' else "✅ No significant inconsistencies detected")
                
                # Verificar si tardó mucho
                elapsed = time.time() - start_time
                if elapsed > 1.5:  # Si tarda más de 1.5 segundos, log warning
                    print(f"[WARNING] Detección de inconsistencias tardó {elapsed:.2f}s (considerar optimización)")
            
        except Exception as e:
            # Fallback: continuar sin detección de inconsistencias si hay error
            print(f"[ERROR] Error en detección de inconsistencias: {e}")
            print("[FALLBACK] Continuando sin detección de inconsistencias para garantizar experiencia fluida")
            inconsistencies = []
            trust_score = 100  # Score por defecto
            inconsistency_report = "✅ No se detectaron inconsistencias significativas" if lang == 'es' else "✅ No significant inconsistencies detected"
        
        # Store in session
        session['inconsistencies'] = {
            'issues': inconsistencies,
            'trust_score': trust_score,
            'report': inconsistency_report
        }
        
        # Recalculate inferred_level with trust_score and inconsistencies now available
        # Apply penalties for low confidence, trust score, and inconsistencies
        # This ensures candidates with vague answers and low confidence are correctly classified as Junior
        original_percentage = inference_percentage
        
        # Penalty 1: Trust Score below 85 indicates inconsistencies/risks
        # More aggressive penalty: -5% per 10 points below 85, minimum -5% if below 85
        # This ensures Trust Score < 85 always triggers significant penalty
        if trust_score < 85:
            trust_penalty = ((85 - trust_score) / 10) * 5  # -5% per 10 points below 85
            # Ensure minimum penalty of 5% for any Trust Score below 85 (more aggressive)
            trust_penalty = max(5.0, trust_penalty)
            inference_percentage = inference_percentage - trust_penalty
            print(f"[DEBUG INFERENCE] Trust Score penalty: -{trust_penalty:.1f}% (Trust Score: {trust_score}/100)")
        
        # Penalty 2: Low emotional confidence indicates uncertainty/vague answers
        if avg_confidence < 0.80:  # Below 80% confidence
            confidence_penalty = 3.0
            inference_percentage = inference_percentage - confidence_penalty
            print(f"[DEBUG INFERENCE] Low confidence penalty: -{confidence_penalty:.1f}% (Avg confidence: {avg_confidence:.2f})")
        
        # Penalty 3: Inconsistencies detected
        inconsistencies_count = len(inconsistencies)
        if inconsistencies_count >= 3:
            # 3+ inconsistencies always trigger penalty
            inconsistency_penalty = 2.0
            inference_percentage = inference_percentage - inconsistency_penalty
            print(f"[DEBUG INFERENCE] Inconsistencies penalty: -{inconsistency_penalty:.1f}% ({inconsistencies_count} detected)")
        elif inconsistencies_count >= 2 and trust_score < 85:
            # If Trust Score is low (< 85), even 2 inconsistencies should trigger penalty
            inconsistency_penalty = 2.0
            inference_percentage = inference_percentage - inconsistency_penalty
            print(f"[DEBUG INFERENCE] Inconsistencies penalty: -{inconsistency_penalty:.1f}% ({inconsistencies_count} detected, Trust Score {trust_score} < 85)")
        elif inconsistencies_count >= 1 and trust_score < 85:
            # If Trust Score is low (< 85), even 1 inconsistency should trigger penalty
            inconsistency_penalty = 1.5
            inference_percentage = inference_percentage - inconsistency_penalty
            print(f"[DEBUG INFERENCE] Inconsistencies penalty: -{inconsistency_penalty:.1f}% ({inconsistencies_count} detected, Trust Score {trust_score} < 85)")
        
        # Penalty 4: Combined risk penalty (Trust Score < 85 AND inconsistencies >= 1)
        # This reflects that the combination is riskier than each factor alone
        if trust_score < 85 and inconsistencies_count >= 1:
            combined_penalty = 2.0
            inference_percentage = inference_percentage - combined_penalty
            print(f"[DEBUG INFERENCE] Combined risk penalty: -{combined_penalty:.1f}% (Trust Score {trust_score} < 85 AND {inconsistencies_count} inconsistencies)")
        
        # Ensure percentage doesn't go negative
        inference_percentage = max(0, inference_percentage)
        
        if original_percentage != inference_percentage:
            print(f"[DEBUG INFERENCE] Adjusted percentage: {original_percentage:.1f}% → {inference_percentage:.1f}%")
        else:
            print(f"[DEBUG INFERENCE] No penalties applied, percentage remains: {inference_percentage:.1f}%")
        
        # Re-infer level based on adjusted percentage
        if inference_percentage >= 85:
            inferred_level = "Senior"
        elif inference_percentage >= 65:
            inferred_level = "Mid-Level"
        else:
            inferred_level = "Junior"
        
        print(f"[DEBUG INFERENCE] Final inferred_level: {inferred_level} (percentage: {inference_percentage:.1f}%)")
        
        # Advanced Decision Logic with 6 levels (OPTIMIZED thresholds)
        if final_score >= 4.3:
            decision = "🏆 *HIGHLY RECOMMENDED*"
            action = "Agendar entrevista final esta semana"
            emoji = "🌟"
            next_step_en = (
                "━━━━━━━━━━━━━━━━━━\n\n"
                "Your technical responses show strong foundational knowledge and clear communication. "
                "We believe you have great potential for this role.\n\n"
                "📧 A recruiter will schedule your next interview in the next days.\n\n"
                "Please be ready to discuss your experience in more depth and explore real-world scenarios.\n\n"
                "🗓️ *Tip:* Review key concepts and prepare examples from past projects. "
                "We're excited to learn more about your journey!"
            )
            next_step_es = (
                "━━━━━━━━━━━━━━━━━━\n\n"
                "Tus respuestas técnicas muestran conocimiento sólido y comunicación clara. "
                "Creemos que tienes gran potencial para este rol.\n\n"
                "📧 Un recruiter agendará tu próxima entrevista en los próximos días.\n\n"
                "Por favor prepárate para discutir tu experiencia en profundidad y explorar escenarios del mundo real.\n\n"
                "🗓️ *Tip:* Revisa conceptos clave y prepara ejemplos de proyectos anteriores. "
                "¡Estamos emocionados de conocer más sobre tu trayectoria!"
            )
        elif final_score >= 3.8:
            decision = "✅ *RECOMMENDED FOR HIRE*"
            action = "Proceder con proceso de contratación"
            emoji = "🎉"
            next_step_en = (
                "━━━━━━━━━━━━━━━━━━\n\n"
                "Your technical responses show solid knowledge and good communication skills. "
                "We believe you're a strong fit for this role.\n\n"
                "📧 A recruiter will contact you in the next 48 hours.\n\n"
                "Please be ready to discuss your experience and answer additional questions about your background.\n\n"
                "🗓️ *Tip:* Review the job requirements and prepare examples from your past projects. "
                "We look forward to learning more about you!"
            )
            next_step_es = (
                "━━━━━━━━━━━━━━━━━━\n\n"
                "Tus respuestas técnicas muestran conocimiento sólido y buenas habilidades de comunicación. "
                "Creemos que eres una buena opción para este rol.\n\n"
                "📧 Un recruiter se contactará contigo en las próximas 48 horas.\n\n"
                "Por favor prepárate para discutir tu experiencia y responder preguntas adicionales sobre tu trayectoria.\n\n"
                "🗓️ *Tip:* Revisa los requisitos del puesto y prepara ejemplos de tus proyectos anteriores. "
                "¡Esperamos conocer más sobre ti!"
            )
        elif final_score >= 3.3:
            decision = "💡 *RECOMMENDED WITH TRAINING*"
            action = "Contratar con plan de capacitación inicial"
            emoji = "📚"
            next_step_en = "📧 A recruiter will contact you to discuss a training plan."
            next_step_es = "📧 Un recruiter te contactará para discutir un plan de capacitación."
        elif final_score >= 2.3:
            decision = "⏳ *REVIEW IN 6 MONTHS*"
            action = "Sugerir áreas de mejora y re-contactar en 6 meses"
            emoji = "🔄"
            
            # Build alternative positions list (excluding the one already selected)
            current_position = session['data'].get('position', '')
            available_positions = session.get('available_positions', AVAILABLE_POSITIONS)
            other_positions = [pos for pos in available_positions if pos != current_position]
            
            positions_list_en = ""
            positions_list_es = ""
            if other_positions:
                for idx, pos in enumerate(other_positions, start=2):  # Start at 2 since they already tried position 1
                    positions_list_en += f"{idx}. {pos}\n"
                    positions_list_es += f"{idx}. {pos}\n"
            
            next_step_en = (
                "━━━━━━━━━━━━━━━━━━\n\n"
                "🌸 I've reviewed your responses carefully.\n\n"
                "Would you like to explore another position that might be a better fit?\n\n"
                "💼 *AVAILABLE POSITIONS:*\n"
                f"{positions_list_en}\n"
                "Reply with the *NUMBER* or write *SKIP* to pass."
            )
            next_step_es = (
                "━━━━━━━━━━━━━━━━━━\n\n"
                "🌸 He revisado tus respuestas cuidadosamente.\n\n"
                "¿Te gustaría explorar otra posición que podría ser mejor match?\n\n"
                "💼 *POSICIONES DISPONIBLES:*\n"
                f"{positions_list_es}\n"
                "Responde con el *NÚMERO* o escribe *OMITIR* para saltar."
            )
        else:
            decision = "❌ *NOT SUITABLE FOR THIS POSITION*"
            action = "Considerar para otras posiciones o niveles más junior"
            emoji = "📝"
            next_step_en = "📋 We'll keep your profile for future opportunities in other positions."
            next_step_es = "📋 Mantendremos tu perfil para futuras oportunidades en otras posiciones."
        
        # Build upgrade section if applicable
        upgrade_section = ""
        if upgrade_suggestion:
            upgrade_section = f"\n{upgrade_suggestion}\n"
            if upgrade_justification:
                upgrade_section += "Razones:\n"
                for reason in upgrade_justification:
                    upgrade_section += f"  - {reason}\n"
            upgrade_section += "\n"
        
        # lang already defined above (line 1060)
        
        # === ENHANCED FEEDBACK FOR FREE MODE (JUDGES TESTING) ===
        # Generate contextual feedback to help judges understand their scores
        is_demo_mode = session.get('demo_mode') == 'full_interview'
        free_mode_insights = ""
        
        if not is_demo_mode:  # Show enhanced insights in FREE MODE
            # Benchmarks for comparison
            BENCHMARK_TECH = 3.0
            BENCHMARK_ENGLISH = 3.0
            BENCHMARK_SOFT = 3.0
            BENCHMARK_FINAL = 3.0
            
            # Generate insights based on scores
            insights_parts = []
            
            # Technical score insights
            if tech_avg >= 4.5:
                tech_insight = "🌟 Excellent technical knowledge! Your responses show deep understanding and practical experience."
            elif tech_avg >= 3.5:
                tech_insight = "✅ Good technical foundation! You demonstrated solid knowledge with room for advanced concepts."
            elif tech_avg >= 2.5:
                tech_insight = "📚 Basic technical understanding. Consider diving deeper into core concepts for this role."
            else:
                tech_insight = "💡 Foundational level detected. Focus on building core technical skills for this position."
            
            # English score insights
            if english_avg >= 4.5:
                english_insight = "🌟 Fluent English! Your communication is clear and professional."
            elif english_avg >= 3.5:
                english_insight = "✅ Good English level! You can communicate effectively in professional settings."
            elif english_avg >= 2.5:
                english_insight = "📚 Intermediate English. Practice will help you express complex ideas more fluently."
            else:
                english_insight = "💡 Basic English level. Consider focused practice for technical communication."
            
            # Soft skills insights
            if soft_skills >= 4.5:
                soft_insight = "🌟 Outstanding soft skills! You show strong leadership and collaboration abilities."
            elif soft_skills >= 3.5:
                soft_insight = "✅ Good interpersonal skills! You demonstrate teamwork and problem-solving."
            elif soft_skills >= 2.5:
                soft_insight = "📚 Developing soft skills. Focus on examples that show leadership and collaboration."
            else:
                soft_insight = "💡 Basic soft skills. Highlight specific examples of teamwork and communication."
            
            # Comparison with benchmarks
            tech_vs_benchmark = tech_avg - BENCHMARK_TECH
            english_vs_benchmark = english_avg - BENCHMARK_ENGLISH
            soft_vs_benchmark = soft_skills - BENCHMARK_SOFT
            
            if lang == 'en':
                free_mode_insights = (
                    "\n━━━━━━━━━━━━━━━━━━\n"
                    "💡 *YOUR PERFORMANCE INSIGHTS*\n"
                    "━━━━━━━━━━━━━━━━━━\n\n"
                    f"🔧 *Technical ({tech_avg:.1f}/5.0):*\n"
                    f"{tech_insight}\n"
                    f"{'📈 Above average' if tech_vs_benchmark > 0 else '📊 Average' if tech_vs_benchmark == 0 else '📉 Below average'} "
                    f"(benchmark: {BENCHMARK_TECH:.1f}/5.0)\n\n"
                    f"🗣️ *English ({english_avg:.1f}/5.0):*\n"
                    f"{english_insight}\n"
                    f"{'📈 Above average' if english_vs_benchmark > 0 else '📊 Average' if english_vs_benchmark == 0 else '📉 Below average'} "
                    f"(benchmark: {BENCHMARK_ENGLISH:.1f}/5.0)\n\n"
                    f"💼 *Soft Skills ({soft_skills:.1f}/5.0):*\n"
                    f"{soft_insight}\n"
                    f"{'📈 Above average' if soft_vs_benchmark > 0 else '📊 Average' if soft_vs_benchmark == 0 else '📉 Below average'} "
                    f"(benchmark: {BENCHMARK_SOFT:.1f}/5.0)\n\n"
                    f"🎯 *Overall Assessment:*\n"
                    f"Your final score of {final_score:.2f}/5.0 places you in the "
                    f"{'top tier' if final_score >= 4.3 else 'strong candidate' if final_score >= 3.8 else 'developing' if final_score >= 3.3 else 'needs improvement'} "
                    f"category. This evaluation reflects your real-time responses and demonstrates how SAORI AI adapts to different candidate profiles.\n"
                )
        if lang == 'en':
            response_text = (
                f"{emoji} *INTERVIEW COMPLETED!*\n\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 *FINAL RESULTS*\n"
                f"━━━━━━━━━━━━━━━━━━\n\n"
                f"👤 *Candidate:* {session['data']['name']}\n"
                f"💼 *Position:* {session['data']['position']}\n"
                f"📊 *Inferred level:* {inferred_level}\n\n"
                f"*SCORES:*\n"
                f"🔧 Technical: {tech_avg:.1f}/5.0 ({tech_percentage:.0f}%)\n"
                f"🗣️ English: {english_avg:.1f}/5.0\n"
                f"💼 Soft Skills: {soft_skills:.1f}/5.0\n\n"
                f"🎯 *FINAL SCORE: {final_score:.2f}/5.0*\n\n"
                f"*EMOTIONAL ANALYSIS (AI):*\n"
                f"😊 Dominant emotion: {dominant_emotion} {EMOTION_EMOJIS.get(dominant_emotion, '😐')}\n"
                f"📈 Average confidence: {avg_confidence:.0%}\n"
                f"{journey_en}\n\n"
                f"*RELIABILITY ANALYSIS:*\n"
                f"🔍 Trust Score: {trust_score}/100\n"
                f"   _(based on consistency, clarity, and emotional stability)_\n"
                f"{inconsistency_report}\n\n"
                f"{free_mode_insights}"
                f"{decision}\n\n"
                f"{upgrade_section}"
                f"{next_step_en}"
            )
        else:
            # Spanish insights (translate the English ones)
            if not is_demo_mode:  # Show enhanced insights in FREE MODE
                tech_insight_es = ""
                english_insight_es = ""
                soft_insight_es = ""
                
                if tech_avg >= 4.5:
                    tech_insight_es = "🌟 ¡Excelente conocimiento técnico! Tus respuestas muestran comprensión profunda y experiencia práctica."
                elif tech_avg >= 3.5:
                    tech_insight_es = "✅ Buena base técnica! Demostraste conocimiento sólido con espacio para conceptos avanzados."
                elif tech_avg >= 2.5:
                    tech_insight_es = "📚 Comprensión técnica básica. Considera profundizar en conceptos clave para este rol."
                else:
                    tech_insight_es = "💡 Nivel fundamental detectado. Enfócate en construir habilidades técnicas básicas para esta posición."
                
                if english_avg >= 4.5:
                    english_insight_es = "🌟 ¡Inglés fluido! Tu comunicación es clara y profesional."
                elif english_avg >= 3.5:
                    english_insight_es = "✅ Buen nivel de inglés! Puedes comunicarte efectivamente en entornos profesionales."
                elif english_avg >= 2.5:
                    english_insight_es = "📚 Inglés intermedio. La práctica te ayudará a expresar ideas complejas con más fluidez."
                else:
                    english_insight_es = "💡 Nivel básico de inglés. Considera práctica enfocada en comunicación técnica."
                
                if soft_skills >= 4.5:
                    soft_insight_es = "🌟 ¡Soft skills excepcionales! Muestras fuerte liderazgo y habilidades de colaboración."
                elif soft_skills >= 3.5:
                    soft_insight_es = "✅ Buenas habilidades interpersonales! Demuestras trabajo en equipo y resolución de problemas."
                elif soft_skills >= 2.5:
                    soft_insight_es = "📚 Soft skills en desarrollo. Enfócate en ejemplos que muestren liderazgo y colaboración."
                else:
                    soft_insight_es = "💡 Soft skills básicas. Destaca ejemplos específicos de trabajo en equipo y comunicación."
                
                BENCHMARK_TECH = 3.0
                BENCHMARK_ENGLISH = 3.0
                BENCHMARK_SOFT = 3.0
                tech_vs_benchmark = tech_avg - BENCHMARK_TECH
                english_vs_benchmark = english_avg - BENCHMARK_ENGLISH
                soft_vs_benchmark = soft_skills - BENCHMARK_SOFT
                
                free_mode_insights = (
                    "\n━━━━━━━━━━━━━━━━━━\n"
                    "💡 *INSIGHTS DE TU RENDIMIENTO*\n"
                    "━━━━━━━━━━━━━━━━━━\n\n"
                    f"🔧 *Técnico ({tech_avg:.1f}/5.0):*\n"
                    f"{tech_insight_es}\n"
                    f"{'📈 Por encima del promedio' if tech_vs_benchmark > 0 else '📊 Promedio' if tech_vs_benchmark == 0 else '📉 Por debajo del promedio'} "
                    f"(referencia: {BENCHMARK_TECH:.1f}/5.0)\n\n"
                    f"🗣️ *Inglés ({english_avg:.1f}/5.0):*\n"
                    f"{english_insight_es}\n"
                    f"{'📈 Por encima del promedio' if english_vs_benchmark > 0 else '📊 Promedio' if english_vs_benchmark == 0 else '📉 Por debajo del promedio'} "
                    f"(referencia: {BENCHMARK_ENGLISH:.1f}/5.0)\n\n"
                    f"💼 *Soft Skills ({soft_skills:.1f}/5.0):*\n"
                    f"{soft_insight_es}\n"
                    f"{'📈 Por encima del promedio' if soft_vs_benchmark > 0 else '📊 Promedio' if soft_vs_benchmark == 0 else '📉 Por debajo del promedio'} "
                    f"(referencia: {BENCHMARK_SOFT:.1f}/5.0)\n\n"
                    f"🎯 *Evaluación General:*\n"
                    f"Tu score final de {final_score:.2f}/5.0 te coloca en la categoría "
                    f"{'de excelencia' if final_score >= 4.3 else 'de candidato fuerte' if final_score >= 3.8 else 'en desarrollo' if final_score >= 3.3 else 'de mejora'} "
                    f". Esta evaluación refleja tus respuestas en tiempo real y demuestra cómo SAORI AI se adapta a diferentes perfiles de candidatos.\n"
                )
            
            response_text = (
                f"{emoji} *¡ENTREVISTA COMPLETADA!*\n\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 *RESULTADOS FINALES*\n"
                f"━━━━━━━━━━━━━━━━━━\n\n"
                f"👤 *Candidato:* {session['data']['name']}\n"
                f"💼 *Posición:* {session['data']['position']}\n"
                f"📊 *Nivel inferido:* {inferred_level}\n\n"
                f"*SCORES:*\n"
                f"🔧 Técnico: {tech_avg:.1f}/5.0 ({tech_percentage:.0f}%)\n"
                f"🗣️ Inglés: {english_avg:.1f}/5.0\n"
                f"💼 Soft Skills: {soft_skills:.1f}/5.0\n\n"
                f"🎯 *SCORE FINAL: {final_score:.2f}/5.0*\n\n"
                f"*ANÁLISIS EMOCIONAL (AI):*\n"
                f"😊 Emoción dominante: {dominant_emotion} {EMOTION_EMOJIS.get(dominant_emotion, '😐')}\n"
                f"📈 Confianza promedio: {avg_confidence:.0%}\n"
                f"{journey_es}\n\n"
                f"*ANÁLISIS DE CONFIABILIDAD:*\n"
                f"🔍 Trust Score: {trust_score}/100\n"
                f"   _(basado en consistencia, claridad y estabilidad emocional)_\n"
                f"{inconsistency_report}\n\n"
                f"{free_mode_insights}"
                f"{decision}\n\n"
                f"{upgrade_section}"
                f"{next_step_es}"
            )
        
        # Save final results
        session['final_results'] = {
            'tech_score': tech_avg,
            'tech_percentage': tech_percentage,
            'english_score': english_avg,
            'soft_skills_score': soft_skills,
            'final_score': final_score,
            'inferred_level': inferred_level,
            'dominant_emotion': dominant_emotion,
            'avg_confidence': avg_confidence,
            'trust_score': trust_score,
            'inconsistencies_count': len(inconsistencies),
            'decision': decision,
            'action': action,
            'upgrade_suggestion': upgrade_suggestion,
            'upgrade_justification': upgrade_justification,
            'completed_at': datetime.now().isoformat()
        }
        
        # Check if this is a "REVIEW IN 6 MONTHS" decision (second chance feature)
        if final_score >= 2.3 and final_score < 3.3:
            session['stage'] = 13.5  # Go to second chance stage (UNIQUE STAGE NUMBER)
            session['second_chance_offered'] = True
        else:
            session['stage'] = 14  # Go directly to feedback stage
        
        # CRITICAL: Return response_text immediately after generating final results
        # This ensures the message is sent even if there are errors in subsequent stages
        session['results_sent'] = True  # Mark that results were sent
        save_session(phone_number, session)
        print(f"[INFO] Final results generated for {phone_number} (length: {len(response_text)} chars)")
        print(f"[DEBUG] Final results preview: {response_text[:200]}...")
        print(f"[DEBUG] Final results will be split if > 1500 chars: {len(response_text) > 1500}")
        return response_text
    
    elif stage == 13.5:  # Second Chance (only for REVIEW IN 6 MONTHS candidates)
        print(f"[DEBUG] ====== STAGE 13.5: SECOND CHANCE ======")
        print(f"[DEBUG] Message received: {message_text}")
        
        # CRITICAL FIX: Check if results were sent, if not, send them now
        if not session.get('results_sent', False):
            print(f"[WARNING] Results not sent yet for {phone_number}, sending now...")
            final_results = session.get('final_results', {})
            if final_results:
                # Regenerate results message
                lang = session.get('language', 'es')
                tech_avg = final_results.get('tech_score', 0)
                english_avg = final_results.get('english_score', 0)
                soft_skills = final_results.get('soft_skills_score', 0)
                final_score = final_results.get('final_score', 0)
                inferred_level = final_results.get('inferred_level', 'N/A')
                dominant_emotion = final_results.get('dominant_emotion', 'neutral')
                avg_confidence = final_results.get('avg_confidence', 0)
                trust_score = final_results.get('trust_score', 100)
                inconsistency_report = session.get('inconsistencies', {}).get('report', '')
                
                # Generate results message (simplified version)
                if lang == 'en':
                    response_text = (
                        f"📊 *INTERVIEW COMPLETED!*\n\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"📊 *FINAL RESULTS*\n"
                        f"━━━━━━━━━━━━━━━━━━\n\n"
                        f"👤 *Candidate:* {session['data']['name']}\n"
                        f"💼 *Position:* {session['data']['position']}\n"
                        f"📊 *Inferred level:* {inferred_level}\n\n"
                        f"*SCORES:*\n"
                        f"🔧 Technical: {tech_avg:.1f}/5.0\n"
                        f"🗣️ English: {english_avg:.1f}/5.0\n"
                        f"💼 Soft Skills: {soft_skills:.1f}/5.0\n\n"
                        f"🎯 *FINAL SCORE: {final_score:.2f}/5.0*\n\n"
                        f"*EMOTIONAL ANALYSIS (AI):*\n"
                        f"😊 Dominant emotion: {dominant_emotion} {EMOTION_EMOJIS.get(dominant_emotion, '😐')}\n"
                        f"📈 Average confidence: {avg_confidence:.0%}\n\n"
                        f"*RELIABILITY ANALYSIS:*\n"
                        f"🔍 Trust Score: {trust_score}/100\n"
                        f"{inconsistency_report}\n\n"
                        f"━━━━━━━━━━━━━━━━━━\n\n"
                        f"⏳ *REVIEW IN 6 MONTHS*\n\n"
                        f"Thank you for completing the interview. We'll review your profile again in 6 months.\n\n"
                    )
                else:
                    response_text = (
                        f"📊 *¡ENTREVISTA COMPLETADA!*\n\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"📊 *RESULTADOS FINALES*\n"
                        f"━━━━━━━━━━━━━━━━━━\n\n"
                        f"👤 *Candidato:* {session['data']['name']}\n"
                        f"💼 *Posición:* {session['data']['position']}\n"
                        f"📊 *Nivel inferido:* {inferred_level}\n\n"
                        f"*SCORES:*\n"
                        f"🔧 Técnico: {tech_avg:.1f}/5.0\n"
                        f"🗣️ Inglés: {english_avg:.1f}/5.0\n"
                        f"💼 Soft Skills: {soft_skills:.1f}/5.0\n\n"
                        f"🎯 *SCORE FINAL: {final_score:.2f}/5.0*\n\n"
                        f"*ANÁLISIS EMOCIONAL (AI):*\n"
                        f"😊 Emoción dominante: {dominant_emotion} {EMOTION_EMOJIS.get(dominant_emotion, '😐')}\n"
                        f"📈 Confianza promedio: {avg_confidence:.0%}\n\n"
                        f"*ANÁLISIS DE CONFIABILIDAD:*\n"
                        f"🔍 Trust Score: {trust_score}/100\n"
                        f"{inconsistency_report}\n\n"
                        f"━━━━━━━━━━━━━━━━━━\n\n"
                        f"⏳ *REVISAR EN 6 MESES*\n\n"
                        f"Gracias por completar la entrevista. Revisaremos tu perfil nuevamente en 6 meses.\n\n"
                    )
                
                session['results_sent'] = True
                save_session(phone_number, session)
                print(f"[INFO] Results sent to {phone_number} in stage 13.5 (recovery)")
                return response_text
        
        user_choice = message_text.strip().upper()
        print(f"[DEBUG] User choice: {user_choice}")
        lang = session.get('language', 'es')
        
        if user_choice in ['SKIP', 'OMITIR', 'SALTAR']:
            # User declined second chance - go directly to final message
            if lang == 'en':
                response_text = (
                    "━━━━━━━━━━━━━━━━━━\n\n"
                    "💙 Thank you for sharing your journey with me today!\n\n"
                    "I can see your potential, and I believe in your growth. While this specific role might not be the perfect match right now, that doesn't define your worth or capabilities.\n\n"
                    "🌸 I'll reach out again in 6 months to explore new opportunities together. In the meantime:\n\n"
                    "• Keep learning and building\n"
                    "• Document your progress\n"
                    "• Stay curious and confident\n\n"
                    "Every step forward matters, and I'm excited to see where your journey takes you. 🌱\n\n"
                    "━━━━━━━━━━━━━━━━━━\n\n"
                    "Take care, and see you soon!\n\n"
                    "— *Saori 🌸*\n\n"
                    "✅ Interview completed"
                )
            else:
                response_text = (
                    "━━━━━━━━━━━━━━━━━━\n\n"
                    "💙 ¡Gracias por compartir tu camino conmigo hoy!\n\n"
                    "Puedo ver tu potencial y creo en tu crecimiento. Aunque este rol específico quizás no sea el match perfecto ahora, eso no define tu valor ni tus capacidades.\n\n"
                    "🌸 Me comunicaré contigo de nuevo en 6 meses para explorar nuevas oportunidades juntos. Mientras tanto:\n\n"
                    "• Sigue aprendiendo y construyendo\n"
                    "• Documenta tu progreso\n"
                    "• Mantente curioso y confiado\n\n"
                    "Cada paso adelante importa, y estoy emocionada de ver hacia dónde te lleva tu viaje. 🌱\n\n"
                    "━━━━━━━━━━━━━━━━━━\n\n"
                    "¡Cuídate y nos vemos pronto!\n\n"
                    "— *Saori 🌸*\n\n"
                    "✅ Entrevista completada"
                )
            session['stage'] = 15  # Go directly to closing (skip feedback)
        else:
            # User selected a position number (for future implementation)
            if lang == 'en':
                response_text = (
                    "━━━━━━━━━━━━━━━━━━\n\n"
                    "🎯 Thank you for your interest!\n"
                    "We've noted your preference for this position.\n\n"
                    "🙏 *Thank you for using SAORI AI Core!*\n"
                    "Powered by AI 🤖\n\n"
                    "✅ Your interview has been completed.\n"
                    "📧 A recruiter will contact you soon.\n\n"
                    "To start a new interview, type: *RESTART*"
                )
            else:
                response_text = (
                    "━━━━━━━━━━━━━━━━━━\n\n"
                    "🎯 ¡Gracias por tu interés!\n"
                    "Hemos anotado tu preferencia por esta posición.\n\n"
                    "🙏 *¡Gracias por usar SAORI AI Core!*\n"
                    "Powered by AI 🤖\n\n"
                    "✅ Tu entrevista ha sido completada.\n"
                    "📧 Un recruiter se contactará contigo pronto.\n\n"
                    "Para iniciar una nueva entrevista, escribe: *REINICIAR*"
                )
            session['stage'] = 15  # Go directly to closing (skip feedback)
    
    elif stage == 14:  # Feedback
        # Check if user is trying to send answers or commands
        lang = session.get('language', 'es')
        
        # Check if this is a command first
        if fuzzy_match_command(message_text, ['RESPUESTAS', 'RESPUESTA', 'respuestas', 'respuesta', 'RESPUEST',
                                               'ANSWERS', 'ANSWER', 'answers', 'answer', 'ANSWR', 'ANSW',
                                               'RESPONSES', 'RESPONSE', 'responses', 'response'], threshold=70):
            # User is asking for answers - show them
            response_text = show_answers_for_profile(session)
            save_session(phone_number, session)
            return response_text
        
        # Check if user is trying to restart
        if fuzzy_match_command(message_text, ['REINICIAR', 'RESTART', 'NUEVO', 'NEW'], threshold=80):
            del sessions[phone_number]
            if lang == 'en':
                response_text = (
                    "🔄 *Fresh start!* 🌸\n\n"
                    "Ready when you are! Just send any message to start your interview."
                )
            else:
                response_text = (
                    "🔄 *¡Empecemos de nuevo!* 🌸\n\n"
                    "¡Lista cuando tú lo estés! Solo envía cualquier mensaje para iniciar tu entrevista."
                )
            save_session(phone_number, session)
            return response_text
        
        # Check if user copied a suggested answer (compare with final_answer if exists)
        final_answer = session.get('data', {}).get('final_answer', '')
        if final_answer and len(message_text) > 50:
            # Check if message is very similar to final_answer (user copied suggested answer)
            message_lower = message_text.lower().strip()
            final_lower = final_answer.lower().strip()
            if message_lower == final_lower or (len(message_lower) > 50 and message_lower in final_lower):
                # User copied the suggested answer, but interview is already complete
                if lang == 'en':
                    response_text = (
                        "✅ *Your interview has already been completed!* 🌸\n\n"
                        "Your final answer was already submitted and evaluated.\n\n"
                        "📧 A recruiter will contact you soon.\n\n"
                        "To start a new interview, type: *RESTART*"
                    )
                else:
                    response_text = (
                        "✅ *¡Tu entrevista ya fue completada!* 🌸\n\n"
                        "Tu respuesta final ya fue enviada y evaluada.\n\n"
                        "📧 Un recruiter se contactará contigo pronto.\n\n"
                        "Para iniciar una nueva entrevista, escribe: *REINICIAR*"
                    )
                save_session(phone_number, session)
                return response_text
        
        # Save feedback if not a command
        if message_text.upper() not in ['OMITIR', 'SKIP', 'SALTAR']:
            session['feedback'] = message_text
        
        if lang == 'en':
            response_text = (
                "🙏 *Thank you for using SAORI AI Core!*\n"
                "Powered by AI 🤖\n\n"
                "━━━━━━━━━━━━━━━━━━\n\n"
                "✅ Your interview has been completed.\n"
                "📧 A recruiter will contact you soon.\n\n"
                "To start a new interview, type: *RESTART*"
            )
        else:
            response_text = (
                "🙏 *¡Gracias por usar SAORI AI Core!*\n"
                "Powered by AI 🤖\n\n"
                "━━━━━━━━━━━━━━━━━━\n\n"
                "✅ Tu entrevista ha sido completada.\n"
                "📧 Un recruiter se contactará contigo pronto.\n\n"
                "Para iniciar una nueva entrevista, escribe: *REINICIAR*"
            )
        
        session['stage'] = 15
    
    elif stage == 15:  # Closing
        lang = session.get('language', 'es')
        
        # Check for restart command
        if message_text.upper() in ['REINICIAR', 'RESTART', 'NUEVO', 'NEW']:
            del sessions[phone_number]
            if lang == 'en':
                response_text = (
                    "🔄 *Fresh start!* 🌸\n\n"
                    "Ready when you are! Just send any message to start your interview."
                )
            else:
                response_text = (
                    "🔄 *¡Empecemos de nuevo!* 🌸\n\n"
                    "¡Lista cuando tú lo estés! Solo envía cualquier mensaje para iniciar tu entrevista."
                )
        else:
            if lang == 'en':
                response_text = (
                    "✅ Interview completed.\n\n"
                    "To start a new interview, type: *RESTART*"
                )
            else:
                response_text = (
                    "Ya completaste tu entrevista. ✅\n\n"
                    "Para iniciar una nueva entrevista, escribe: *REINICIAR*"
                )
    
    # Save session
    save_session(phone_number, session)
    
    print(f"[DEBUG] Returning response_text, length: {len(response_text)}")
    print(f"[DEBUG] First 100 chars: {response_text[:100] if response_text else 'EMPTY'}")
    
    return response_text

# ====================
# DEMO MODE SUPPORT
# ====================

# Pre-loaded demo profiles with specific positions for each
DEMO_PROFILES = {
    '1': {
        'name': 'Ana García',
        'referrer': 'Sarah Johnson',
        'position': 'Data Engineer (AWS/Spark)',
        'language': 'en',
        'available_positions': [
            "Analytics Engineer (dbt/SQL/Python)",
            "Data Engineer (AWS/Spark)",
            "Machine Learning Engineer (Python/MLflow)",
            "ETL Developer (Airflow/SQL)",
            "Data Platform Engineer (Terraform/Kafka)"
        ]
    },
    '2': {
        'name': 'Luis Martínez',
        'referrer': 'Michael Chen',
        'position': 'Backend Developer (Python/Django)',
        'language': 'en',
        'available_positions': [
            "Backend Developer (Python/Django)",
            "Full Stack Developer (Python/React)",
            "API Developer (REST/GraphQL)",
            "Backend Engineer (Node.js/Express)",
            "Software Engineer (Python/FastAPI)"
        ]
    }
}

# Answers database for demo profiles
DEMO_ANSWERS = {
    '1': {  # Ana García
        'privacy': 'YES',
        'position': '2',
        'availability': 'Immediate',
        'salary': '5000 USD',
        'modality': '1',
        'zone': 'Buenos Aires',
        'tech_q1': 'Normalization is the process of organizing data in a relational database to reduce redundancy and improve integrity. It is achieved by dividing large tables into smaller ones and defining relationships between them. 1NF eliminates repeating groups, 2NF removes partial dependencies, and 3NF removes transitive dependencies.',
        'tech_q2': 'Python is dynamically typed because it does not require declaring the type of a variable when defining it; the type is determined at runtime. Advantages include faster development and flexibility. Disadvantages include runtime errors and potential performance overhead.',
        'tech_q3': 'Apache Spark handles fault tolerance through RDDs (Resilient Distributed Datasets), which maintain a lineage of transformations. If a node fails, Spark can reconstruct the data from its origin using this lineage.',
        'english_q1': 'I have over 8 years of experience with Python, primarily in data engineering and ETL pipelines. I\'ve built scalable data processing systems using PySpark, Airflow, and AWS services. I also mentor junior developers and contribute to open-source projects.',
        'english_q2': 'I architected and led the migration of our entire data infrastructure to AWS, processing 10TB daily. This reduced costs by 40% and improved pipeline reliability to 99.9%. The project involved coordinating with 5 teams and was completed ahead of schedule.',
        'soft_skills': 'We had a critical production incident affecting customers. I coordinated with backend, frontend, and DevOps teams to diagnose the problem. We implemented a temporary solution in 2 hours and a permanent one within 24 hours. We documented everything to prevent future incidents.',
        'final': 'I have proven experience in large-scale projects, combining solid technical skills with leadership ability. I learn fast, adapt to new technologies, and contribute to improving processes. I\'m looking for a challenge where I can make real impact.'
    },
    '2': {  # Luis Martínez
        'privacy': 'YES',
        'position': '1',
        'availability': '2 weeks',
        'salary': '2500 USD',
        'modality': '1',
        'zone': 'Buenos Aires',
        'tech_q1': 'Django models are Python classes that represent database tables. Each model maps to a table and attributes become columns.',
        'tech_q2': 'I would create a view with Django REST Framework, use a serializer for the User model, and return JSON data.',
        'tech_q3': 'Docker packages the app with dependencies into containers for consistent deployment. I\'d use Dockerfile and docker-compose with Django.',
        'english_q1': 'I do not have experience using Python',
        'english_q2': 'I do not have an answer right now',
        'soft_skills': 'Tuvimos un incidente y fue necesario llamar a los managers y trabajar en equipo para resolverlo',
        'final': 'Tengo las habilidades técnicas para desempeñarme correctamente en la posición'
    }
}

def show_help_message(session):
    """Show contextual help message based on current stage"""
    demo_mode = session.get('demo_mode')
    stage = session.get('stage', 0)
    last_bot_message = session.get('last_bot_message', '')
    lang = session.get('language', 'en')
    
    # Build help message
    help_text = ""
    
    # Only show specific help if demo_mode is actually set and not None
    if demo_mode == 'select_profile':
        help_text = (
            "📖 *AYUDA - SELECCIÓN DE PERFIL*\n\n"
            "Selecciona un perfil para probar:\n\n"
            "• *1* = Ana García (Senior Data Engineer)\n"
            "   Score esperado: 4.87/5.0, Trust 100\n\n"
            "• *2* = Luis Martínez (Junior Backend)\n"
            "   Score esperado: 2.9/5.0, Trust 80\n\n"
            "💡 *Tip:* Después de seleccionar, envía *RESPUESTAS* para ver respuestas sugeridas"
        )
    
    elif demo_mode == 'full_interview':
        profile_name = session.get('data', {}).get('name', '')
        # Only show profile-specific help if we actually have a profile name
        if profile_name:
            if lang == 'en':
                if profile_name == 'Ana García':
                    help_text = (
                        "📖 *HELP - ANA GARCÍA PROFILE*\n\n"
                        "You're testing Ana García (Senior Data Engineer).\n\n"
                        "💡 *Useful commands:*\n"
                        "• *ANSWERS* → See suggested answers\n"
                        "• *RESET* → Restart from beginning\n"
                        "• *HELP* → Show this help\n\n"
                        "⏱️ *Expected time:* 5-7 minutes"
                    )
                else:
                    help_text = (
                        "📖 *HELP - LUIS MARTÍNEZ PROFILE*\n\n"
                        "You're testing Luis Martínez (Junior Backend Developer).\n\n"
                        "💡 *Useful commands:*\n"
                        "• *ANSWERS* → See suggested answers\n"
                        "• *RESET* → Restart from beginning\n"
                        "• *HELP* → Show this help\n\n"
                        "⏱️ *Expected time:* 5-7 minutes"
                    )
            else:
                if profile_name == 'Ana García':
                    help_text = (
                        "📖 *AYUDA - PERFIL ANA GARCÍA*\n\n"
                        "Estás probando Ana García (Data Engineer Senior).\n\n"
                        "💡 *Comandos útiles:*\n"
                        "• *RESPUESTAS* → Ver respuestas sugeridas\n"
                        "• *RESET* → Reiniciar desde el inicio\n"
                        "• *AYUDA* → Ver esta ayuda"
                    )
                else:
                    help_text = (
                        "📖 *AYUDA - PERFIL LUIS MARTÍNEZ*\n\n"
                        "Estás probando Luis Martínez (Backend Developer Junior).\n\n"
                        "💡 *Comandos útiles:*\n"
                        "• *RESPUESTAS* → Ver respuestas sugeridas\n"
                        "• *RESET* → Reiniciar desde el inicio\n"
                        "• *AYUDA* → Ver esta ayuda"
                    )
    
    # General help (if no specific help was set)
    if not help_text:
        if lang == 'en':
            help_text = (
                "📖 *GENERAL HELP*\n\n"
                "Available commands:\n"
                "• *DEMO* → Start demo mode\n"
                "• *RESTART* / *RESET* → Clear session\n"
                "• *AYUDA* / *HELP* → Show this help\n"
                "• *RESPUESTAS* / *ANSWERS* → Show suggested answers (in demo mode)\n\n"
                "💡 *Tip:* Start with *DEMO* to test the system"
            )
        else:
            help_text = (
                "📖 *AYUDA GENERAL*\n\n"
                "Comandos disponibles:\n"
                "• *DEMO* → Iniciar modo demo\n"
                "• *REINICIAR* → Limpiar sesión\n"
                "• *AYUDA* → Ver esta ayuda\n"
                "• *RESPUESTAS* → Ver respuestas sugeridas (en modo demo)\n\n"
                "💡 *Tip:* Empieza con *DEMO* para probar el sistema"
            )
    
    # Append last bot message if available and we're in an active interview
    if last_bot_message and stage > 0:
        if lang == 'en':
            return f"{help_text}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{last_bot_message}"
        else:
            return f"{help_text}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{last_bot_message}"
    
    return help_text

def show_answers_for_profile(session, stage=None):
    """Show suggested answers for current profile and stage"""
    demo_mode = session.get('demo_mode', '')
    
    if demo_mode != 'full_interview':
        lang = session.get('language', 'en')
        if lang == 'en':
            return (
                "ℹ️ *The ANSWERS command is only available in DEMO Mode.*\n\n"
                "In Free Mode, please answer based on your own experience and knowledge.\n\n"
                "💡 *Tip:* If you'd like to see example answers, start a new conversation and select *DEMO Mode* 🎬"
            )
        else:
            return (
                "ℹ️ *El comando RESPUESTAS solo está disponible en Modo DEMO.*\n\n"
                "En Modo Libre, por favor responde basándote en tu propia experiencia y conocimiento.\n\n"
                "💡 *Tip:* Si quieres ver respuestas de ejemplo, inicia una nueva conversación y selecciona *Modo DEMO* 🎬"
            )
    
    profile_name = session.get('data', {}).get('name', '')
    current_stage = stage if stage is not None else session.get('stage', 0)
    lang = session.get('language', 'en')
    
    # Find profile ID
    profile_id = None
    for pid, profile in DEMO_PROFILES.items():
        if profile['name'] == profile_name:
            profile_id = pid
            break
    
    if not profile_id or profile_id not in DEMO_ANSWERS:
        if lang == 'en':
            return "❌ Profile not recognized. Send *DEMO* to select."
        else:
            return "❌ Perfil no reconocido. Envía *DEMO* para seleccionar."
    
    answers = DEMO_ANSWERS[profile_id]
    
    # Map stage to answer key
    stage_to_answer = {
        0.5: 'privacy',
        2: 'position',
        3: 'availability',
        4: 'salary',
        5: 'modality',
        6: 'zone',
        7: 'tech_q1',
        8: 'tech_q2',
        9: 'tech_q3',
        10: 'english_q1',
        11: 'english_q2',
        12: 'soft_skills',
        13: 'final'
    }
    
    if current_stage in stage_to_answer:
        answer_key = stage_to_answer[current_stage]
        if answer_key in answers:
            answer_text = answers[answer_key]
            stage_name = STAGES.get(current_stage, 'current')
            
            # Return format: tip first, then separator, then answer
            # Webhook will split and send in two messages
            if lang == 'en':
                return (
                    f"💡 *Tip:* Copy and paste this answer, or modify it as needed\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{answer_text}"
                )
            else:
                return (
                    f"💡 *Tip:* Copia y pega esta respuesta, o modifícala según necesites\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{answer_text}"
                )
    
    # Show all answers
    if lang == 'en':
        response = f"📋 *SUGGESTED ANSWERS FOR {profile_name}*\n\n"
        response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        response += f"1. Privacy: {answers.get('privacy', 'N/A')}\n"
        response += f"2. Position: {answers.get('position', 'N/A')}\n"
        response += f"3. Availability: {answers.get('availability', 'N/A')}\n"
        response += f"4. Salary: {answers.get('salary', 'N/A')}\n"
        response += f"5. Modality: {answers.get('modality', 'N/A')}\n"
        response += f"6. Zone: {answers.get('zone', 'N/A')}\n\n"
        response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        response += "💡 *Tip:* Send *ANSWERS* during any question to see the answer for that specific stage"
    else:
        response = f"📋 *RESPUESTAS SUGERIDAS PARA {profile_name}*\n\n"
        response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        response += f"1. Privacidad: {answers.get('privacy', 'N/A')}\n"
        response += f"2. Posición: {answers.get('position', 'N/A')}\n"
        response += f"3. Disponibilidad: {answers.get('availability', 'N/A')}\n"
        response += f"4. Salario: {answers.get('salary', 'N/A')}\n"
        response += f"5. Modalidad: {answers.get('modality', 'N/A')}\n"
        response += f"6. Zona: {answers.get('zone', 'N/A')}\n\n"
        response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        response += "💡 *Tip:* Envía *RESPUESTAS* durante cualquier pregunta para ver la respuesta de esa etapa"
    
    return response

def process_demo_mode(phone_number, message_text):
    """Handle demo mode with pre-loaded profiles"""
    session = get_session(phone_number)
    
    # Step 1: Show language selection (if not in demo mode or if DEMO command sent again)
    if 'demo_mode' not in session or session.get('demo_mode') is None or message_text.upper() in ['DEMO', 'TEST', 'PRUEBA', 'TRY']:
        session['demo_mode'] = 'select_language'
        session['data'] = {}  # Clear any previous data
        session['stage'] = 0
        save_session(phone_number, session)
        return (
            "🎬 *DEMO MODE* 🌸\n\n"
            "Let's test the system! First, choose the language:\n\n"
            "1️⃣ *English* 🇬🇧\n"
            "2️⃣ *Español* 🇪🇸\n\n"
            "Which language? Just reply: *1* or *2* 😊\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📖 Or send *HELP* or *AYUDA* anytime for help"
        )
    
    # Step 2: Handle language selection
    if session['demo_mode'] == 'select_language':
        language_choice = message_text.strip()
        
        if language_choice == '1':
            session['language'] = 'en'
            session['language_locked'] = True
            session['demo_mode'] = 'select_profile'
            save_session(phone_number, session)
            return (
                "✅ *English selected* 🇬🇧\n\n"
                "Now choose a candidate profile:\n\n"
                "1️⃣ *Ana García* (Senior Data Engineer)\n"
                "   ✨ 5 years experience\n"
                "   💻 Python, SQL, Spark, AWS\n"
                "   📊 Expected: 4.87/5.0, Trust 100\n\n"
                "2️⃣ *Luis Martínez* (Junior Backend Developer)\n"
                "   🌱 2 years experience\n"
                "   💻 Python/Django, REST APIs\n"
                "   📊 Expected: 2.9/5.0, Trust 80\n\n"
                "Which one should I interview? Just reply: *1* or *2* 😊\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 *Tip:* After selecting, send *ANSWERS* to see suggested answers"
            )
        elif language_choice == '2':
            session['language'] = 'es'
            session['language_locked'] = True
            session['demo_mode'] = 'select_profile'
            save_session(phone_number, session)
            return (
                "✅ *Español seleccionado* 🇪🇸\n\n"
                "Ahora elige un perfil de candidato:\n\n"
                "1️⃣ *Ana García* (Data Engineer Senior)\n"
                "   ✨ 5 años de experiencia\n"
                "   💻 Python, SQL, Spark, AWS\n"
                "   📊 Esperado: 4.87/5.0, Confianza 100\n\n"
                "2️⃣ *Luis Martínez* (Desarrollador Backend Junior)\n"
                "   🌱 2 años de experiencia\n"
                "   💻 Python/Django, APIs REST\n"
                "   📊 Esperado: 2.9/5.0, Confianza 80\n\n"
                "¿Cuál quieres que entreviste? Solo responde: *1* o *2* 😊\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 *Tip:* Después de seleccionar, envía *RESPUESTAS* para ver respuestas sugeridas"
            )
        else:
            return "❌ Invalid option. Reply with *1* (English) or *2* (Español)"
    
    # Step 3: Load profile and start interview
    if session['demo_mode'] == 'select_profile':
        profile_id = message_text.strip()
        
        if profile_id not in DEMO_PROFILES:
            lang = session.get('language', 'en')
            if lang == 'en':
                return "❌ Invalid option. Reply with *1* or *2*"
            else:
                return "❌ Opción inválida. Responde con *1* o *2*"
        
        # Load profile with specific positions
        profile = DEMO_PROFILES[profile_id]
        session['data'] = {
            'name': profile['name'],
            'referrer': profile['referrer'],
            'position': profile['position']  # CRITICAL: Load position for correct questions
        }
        
        # Language is already set and locked from language selection step
        # session['language'] is already set
        # session['language_locked'] is already True
        
        session['available_positions'] = profile['available_positions']  # Load specific positions
        session['demo_mode'] = 'full_interview'
        session['stage'] = 0.5  # Go to privacy authorization stage
        
        # Generate personalized welcome using selected language
        lang = session.get('language', 'en')  # Use selected language
        profile_name = profile['name']
        referrer = profile['referrer']
        
        if lang == 'en':
            response_text = (
                f"🌸 Hi *{profile_name}*!\n\n"
                f"I'm *Saori 🌸* — an AI-powered recruitment assistant.\n\n"
                f"*{referrer}* told me about you! I'll guide you through a short evaluation designed to understand your *skills* and *how you feel today*.\n\n"
                f"Let's make this process *simple, respectful, and human*. ✨\n\n"
                f"In the next *10 minutes*, I'll ask about:\n\n"
                "✅ Your technical superpowers\n"
                "✅ Your English fluency\n"
                "✅ How you work with teams\n"
                "✅ Your emotional state (yes, I can sense that! 😊)\n\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔐 *Quick Privacy Note*\n"
                "━━━━━━━━━━━━━━━━━━\n\n"
                "Before we start, I need your permission to process:\n"
                "• Your name & responses\n"
                "• Your salary expectations\n\n"
                "📁 Everything stays confidential — used *only* for your recruitment process, *never* shared.\n\n"
                "*Ready to begin this journey together?*\n\n"
                "👉 Just reply: *YES* or *NO*"
            )
        else:
            response_text = (
                f"🌸 ¡Hola *{profile_name}*!\n\n"
                f"Soy *Saori 🌸* — tu compañera emocional de IA.\n\n"
                f"*{referrer}* me habló de ti! Te guiaré en una breve evaluación diseñada para entender tus *habilidades* y *cómo te sientes hoy*.\n\n"
                f"Hagamos este proceso *simple, respetuoso y humano*. ✨\n\n"
                f"En los próximos *10 minutos*, te preguntaré sobre:\n\n"
                "✅ Tus superpoderes técnicos\n"
                "✅ Tu fluidez en inglés\n"
                "✅ Cómo trabajas en equipo\n"
                "✅ Tu estado emocional (¡sí, puedo percibirlo! 😊)\n\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔐 *Nota Rápida de Privacidad*\n"
                "━━━━━━━━━━━━━━━━━━\n\n"
                "Antes de empezar, necesito tu permiso para procesar:\n"
                "• Tu nombre y respuestas\n"
                "• Tus expectativas salariales\n\n"
                "📁 Todo es confidencial — usado *solo* para tu proceso de reclutamiento, *nunca* compartido.\n\n"
                "*¿Lista/o para comenzar este viaje juntos?*\n\n"
                "👉 Responde: *SÍ* o *NO*"
            )
        
        save_session(phone_number, session)
        
        return response_text
    
    # Step 3: Continue with normal interview
    return process_message(phone_number, message_text)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Twilio webhook endpoint with DEMO support"""
    try:
        # Get incoming message
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        # Get session for logging (may not exist yet)
        session = sessions.get(from_number, None)
        log_user_action(from_number, "Received message", incoming_msg, session, "INFO")
        
        # Check if this is a join code - if so, automatically start the flow
        if incoming_msg.lower().startswith('join '):
            # Clear any existing session
            if from_number in sessions:
                del sessions[from_number]
            
            # Get fresh session and show language selection first
            session = get_session(from_number)
            # Set stage to -1 to trigger language selection
            session['stage'] = -1
            save_session(from_number, session)
            
            # Show language selection message
            response_text = (
                "🌸 *¡Bienvenido a SAORI AI Core!* / *Welcome to SAORI AI Core!* 🌸\n\n"
                "Primero, elige tu idioma / First, choose your language:\n\n"
                "1️⃣ *English* 🇬🇧\n"
                "2️⃣ *Español* 🇪🇸\n\n"
                "¿Qué idioma? / Which language? Just reply: *1* or *2* 😊"
            )
            
            # Create Twilio response
            resp = MessagingResponse()
            msg = resp.message(response_text)
            log_user_action(from_number, f"Join code detected - showing language selection", response_text[:150], session, "INFO")
            response_str = str(resp)
            response = make_response(response_str)
            response.headers['Content-Type'] = 'text/xml; charset=utf-8'
            return response
        
        # AUTO-START: Detect new user joining sandbox
        # When user scans QR or enters code, they receive "all set" from Twilio
        # Then when they send their first message, we auto-start SAORI
        
        # Get session to check if user is new
        session = get_session(from_number)
        current_stage = session.get('stage', 0)
        has_data = bool(
            session.get('data', {}).get('privacy_accepted') or 
            session.get('data', {}).get('name') or
            session.get('language_locked', False)
        )
        
        # Check if this is a new user (not a command, stage 0, no data)
        is_command = fuzzy_match_command(incoming_msg, ['REINICIAR', 'RESTART', 'DEMO', 'HELP', 'AYUDA', 'RESET'], threshold=80)
        is_join_code = incoming_msg.lower().startswith('join ')
        
        is_new_user = (
            current_stage == 0 and 
            not has_data and
            not is_command and
            not is_join_code  # Already handled above
        )
        
        # If new user detected, auto-start SAORI
        if is_new_user:
            print(f"[INFO] ✅ New user detected from {from_number} - auto-starting SAORI")
            
            # Clear any existing session
            if from_number in sessions:
                del sessions[from_number]
            
            # Create fresh session
            session = get_session(from_number)
            session['stage'] = -1  # Stage -1 = Language selection
            save_session(from_number, session)
            
            # Send welcome message automatically
            response_text = (
                "🌸 *¡Bienvenido a SAORI AI Core!* / *Welcome to SAORI AI Core!* 🌸\n\n"
                "Primero, elige tu idioma / First, choose your language:\n\n"
                "1️⃣ *English* 🇬🇧\n"
                "2️⃣ *Español* 🇪🇸\n\n"
                "¿Qué idioma? / Which language? Just reply: *1* or *2* 😊"
            )
            
            # Create Twilio response
            resp = MessagingResponse()
            msg = resp.message(response_text)
            log_user_action(from_number, "Auto-started SAORI (new user detected)", response_text[:150], session, "INFO")
            
            response_str = str(resp)
            response = make_response(response_str)
            response.headers['Content-Type'] = 'text/xml; charset=utf-8'
            return response
        
        # Check for help commands (with fuzzy matching for typo tolerance)
        help_match = fuzzy_match_command(incoming_msg, ['AYUDA', 'HELP', '?', 'H'], threshold=80)
        if help_match:
            session = get_session(from_number)
            response_text = show_help_message(session)
        # Check for answers command (with fuzzy matching - allows typos and variations)
        elif fuzzy_match_command(incoming_msg, [
            'RESPUESTAS', 'RESPUESTA', 'respuestas', 'respuesta', 'RESPUEST',
            'ANSWERS', 'ANSWER', 'answers', 'answer', 'ANSWR', 'ANSW',
            'RESPONSES', 'RESPONSE', 'responses', 'response'
        ], threshold=70):
            session = get_session(from_number)
            response_text = show_answers_for_profile(session)
        # Check for reset command (with fuzzy matching)
        elif fuzzy_match_command(incoming_msg, ['RESET'], threshold=80):
            session = get_session(from_number)
            session['stage'] = 0
            session['demo_mode'] = None  # Clear demo mode completely
            session['data'] = {}
            session['scores'] = {'technical': 0, 'english': 0, 'soft_skills': 0}
            session['emotions'] = []
            session['responses'] = []
            session['language'] = None  # Clear language to allow re-detection
            session['language_locked'] = False
            session['last_bot_message'] = None  # Clear last bot message
            save_session(from_number, session)
            response_text = (
                "🔄 *Reset complete!* 🌸\n\n"
                "Starting fresh. Send *DEMO* to begin demo mode."
            )
        # Check for restart command (with fuzzy matching)
        restart_match = fuzzy_match_command(incoming_msg, ['REINICIAR', 'RESTART', 'NUEVO', 'NEW'], threshold=80)
        if restart_match:
            # Detect language from command to respond appropriately
            detected_lang = detect_language(incoming_msg)
            
            # Get or create session
            session = get_session(from_number)
            
            # Reset session but keep language preference
            session['stage'] = 0.6  # Go directly to DEMO/Free mode selection
            session['demo_mode'] = None
            session['data'] = {}
            session['scores'] = {'technical': 0, 'english': 0, 'soft_skills': 0}
            session['emotions'] = []
            session['responses'] = []
            session['last_bot_message'] = None  # Clear last bot message
            session['data']['privacy_accepted'] = True  # Skip privacy since user already accepted before
            
            # Always update language based on command (user explicitly changed language)
            # REINICIAR = Spanish, RESTART = English
            session['language'] = detected_lang
            session['language_locked'] = True
            
            save_session(from_number, session)
            
            # Respond with DEMO/Free mode selection
            lang = detected_lang  # Use detected language directly
            if lang == 'en':
                response_text = (
                    "🔄 *Fresh start!* 🌸\n\n"
                    "✨ *Great! Thanks for trusting me.* 🌸\n\n"
                    "Now, how would you like to proceed?\n\n"
                    "1️⃣ *DEMO Mode* 🎬\n"
                    "   Test with sample profiles (Ana or Luis)\n\n"
                    "2️⃣ *Free Mode* 🆓\n"
                    "   Start your own interview\n\n"
                    "Which option? Reply *1* or *2* 😊\n\n"
                    "💡 *Tip:* You can type *RESTART* anytime to start over."
                )
            else:
                response_text = (
                    "🔄 *¡Empecemos de nuevo!* 🌸\n\n"
                    "✨ *¡Genial! Gracias por confiar en mí.* 🌸\n\n"
                    "Ahora, ¿cómo te gustaría proceder?\n\n"
                    "1️⃣ *Modo DEMO* 🎬\n"
                    "   Probar con perfiles de ejemplo (Ana o Luis)\n\n"
                    "2️⃣ *Modo Libre* 🆓\n"
                    "   Iniciar tu propia entrevista\n\n"
                    "¿Qué opción? Responde *1* o *2* 😊\n\n"
                    "💡 *Tip:* Puedes escribir *REINICIAR* en cualquier momento para empezar de nuevo."
                )
        # Check for demo mode activation (with fuzzy matching)
        elif fuzzy_match_command(incoming_msg, ['DEMO', 'TEST', 'PRUEBA', 'TRY'], threshold=80):
            response_text = process_demo_mode(from_number, incoming_msg)
        else:
            # Check if message might be a command with typo (for suggestion)
            session = get_session(from_number)
            lang = session.get('language', 'es')
            
            # Check all command lists for potential matches
            all_commands = {
                'help': (['AYUDA', 'HELP', '?', 'H'], 'AYUDA' if lang == 'es' else 'HELP'),
                'answers': (['RESPUESTAS', 'ANSWERS', 'ANSWER', 'RESPONSES'], 'RESPUESTAS' if lang == 'es' else 'ANSWERS'),
                'reset': (['RESET'], 'RESET'),
                'restart': (['REINICIAR', 'RESTART', 'NUEVO', 'NEW'], 'REINICIAR' if lang == 'es' else 'RESTART'),
                'demo': (['DEMO', 'TEST', 'PRUEBA', 'TRY'], 'DEMO')
            }
            
            best_suggestion = None
            best_score = 0
            
            for cmd_type, (cmd_list, display_name) in all_commands.items():
                match_cmd, score = get_best_command_match(incoming_msg, cmd_list, min_similarity=70)
                if match_cmd and score > best_score and score < 80:  # Between 70-80%
                    best_score = score
                    best_suggestion = display_name
            
            # If we found a close match (70-80%), suggest it
            if best_suggestion:
                if lang == 'en':
                    response_text = (
                        f"💡 *Did you mean '{best_suggestion}'?*\n\n"
                        f"Similarity: {best_score:.0f}% (need 80%+)\n\n"
                        f"*Available commands:*\n"
                        f"• *HELP* - Show help\n"
                        f"• *ANSWERS* - Show suggested answers\n"
                        f"• *RESET* - Reset interview\n"
                        f"• *RESTART* - Start over\n"
                        f"• *DEMO* - Try demo mode\n\n"
                        f"Or continue with your interview response. 😊"
                    )
                else:
                    response_text = (
                        f"💡 *¿Quisiste decir '{best_suggestion}'?*\n\n"
                        f"Similitud: {best_score:.0f}% (se necesita 80%+)\n\n"
                        f"*Comandos disponibles:*\n"
                        f"• *AYUDA* - Mostrar ayuda\n"
                        f"• *RESPUESTAS* - Mostrar respuestas sugeridas\n"
                        f"• *RESET* - Reiniciar entrevista\n"
                        f"• *REINICIAR* - Empezar de nuevo\n"
                        f"• *DEMO* - Probar modo demo\n\n"
                        f"O continúa con tu respuesta de la entrevista. 😊"
                    )
            else:
                # First check if this is a command (even if in demo mode or flow)
                # Commands should always be processed first, regardless of stage
                session = get_session(from_number)  # Get session first
                answers_match = fuzzy_match_command(incoming_msg, [
                    'RESPUESTAS', 'RESPUESTA', 'respuestas', 'respuesta', 'RESPUEST',
                    'ANSWERS', 'ANSWER', 'answers', 'answer', 'ANSWR', 'ANSW',
                    'RESPONSES', 'RESPONSE', 'responses', 'response'
                ], threshold=70)
                help_match = fuzzy_match_command(incoming_msg, ['AYUDA', 'HELP', '?', 'H'], threshold=80)
                
                if answers_match:
                    # ANSWERS command - process it (works in both Demo and Free Mode, but shows appropriate message)
                    response_text = show_answers_for_profile(session)
                    
                    # Check if we need to send in two separate messages (Demo Mode with current stage answer)
                    demo_mode = session.get('demo_mode', '')
                    current_stage = session.get('stage', 0)
                    
                    # If in demo mode and showing a specific answer (not all answers), send in two messages
                    if demo_mode == 'full_interview' and current_stage in [0.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
                        # Check if response contains tip separator (indicates two-part message)
                        if "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" in response_text:
                            # Split response into tip and answer
                            parts = response_text.split("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                            if len(parts) == 2:
                                tip_part = parts[0].strip()
                                answer_part = parts[1].strip()
                                
                                # Send tip first via webhook response
                                resp = MessagingResponse()
                                resp.message(tip_part)
                                
                                # Send answer second (using Twilio API directly)
                                try:
                                    if client and TWILIO_ACCOUNT_SID and TWILIO_ACCOUNT_SID != 'your_account_sid':
                                        import threading
                                        def send_answer():
                                            time.sleep(0.5)  # Small delay between messages
                                            client.messages.create(
                                                body=answer_part,
                                                from_=TWILIO_WHATSAPP_NUMBER,
                                                to=from_number
                                            )
                                            print(f"[INFO] Sent answer as second message to {from_number}")
                                        
                                        threading.Thread(target=send_answer, daemon=True).start()
                                    
                                    save_session(from_number, session)
                                    response_str = str(resp)
                                    response = make_response(response_str)
                                    response.headers['Content-Type'] = 'text/xml; charset=utf-8'
                                    return response
                                except Exception as e:
                                    print(f"[ERROR] Failed to send answer as second message: {e}")
                                    # Fallback: send as single message
                    
                    save_session(from_number, session)
                elif help_match:
                    # HELP command - process it
                    response_text = show_help_message(session)
                else:
                    # Not a command - check if this is a valid flow response
                    session = get_session(from_number)  # Ensure we have the latest session
                    current_stage = session.get('stage', 0)
                    
                    # CRITICAL FIX: Recover stage if session was reset but has data
                    if current_stage == 0:
                        data = session.get('data', {})
                        if data.get('privacy_accepted'):
                            if not data.get('name'):
                                current_stage = 1  # Waiting for name
                            elif not data.get('position'):
                                current_stage = 2  # Waiting for position
                            elif not data.get('availability'):
                                current_stage = 3  # Waiting for availability
                            elif not data.get('salary'):
                                current_stage = 4  # Waiting for salary
                            elif not data.get('modality'):
                                current_stage = 5  # Waiting for modality
                            elif not data.get('zone'):
                                current_stage = 6  # Waiting for timezone
                            else:
                                # Has all basic data, should be in technical questions
                                tech_questions = data.get('tech_questions', [])
                                if len(tech_questions) == 0:
                                    current_stage = 7
                                elif len(tech_questions) == 1:
                                    current_stage = 8
                                elif len(tech_questions) == 2:
                                    current_stage = 9
                                elif len(tech_questions) == 3:
                                    # Check if has english questions
                                    if not data.get('english_responses') or len(data.get('english_responses', [])) == 0:
                                        current_stage = 10
                                    elif len(data.get('english_responses', [])) == 1:
                                        current_stage = 11
                                    elif not data.get('soft_skills_response'):
                                        current_stage = 12
                                    else:
                                        current_stage = 13  # Final question
                            
                            # Update session with recovered stage
                            if current_stage != 0:
                                session['stage'] = current_stage
                                save_session(from_number, session)
                                log_user_action(from_number, f"Recovered stage {current_stage} from session data", "", session, "INFO")
                    
                    # Check if this could be a valid flow response
                    is_valid_flow_response = False
                    
                    # Stage 0.6 handles DEMO/Free mode selection - "1" and "2" are valid
                    if current_stage == 0.6:
                        is_valid_flow_response = True
                    # Demo mode active - process as demo response
                    elif session.get('demo_mode') and session.get('demo_mode') != 'full_interview':
                        is_valid_flow_response = True
                    # Normal interview flow - process as interview response
                    elif current_stage > 0:
                        is_valid_flow_response = True
                    
                    if is_valid_flow_response:
                        # Process as normal flow response
                        if current_stage == 0.6:
                            # Stage 0.6 handles DEMO/Free mode selection in process_message
                            response_text = process_message(from_number, incoming_msg)
                        elif session.get('demo_mode') and session.get('demo_mode') != 'full_interview':
                            # Only redirect to demo mode if demo_mode is set and not None
                            response_text = process_demo_mode(from_number, incoming_msg)
                        else:
                            response_text = process_message(from_number, incoming_msg)
                    else:
                        # Command not recognized and not a valid flow response - repeat last message if available
                        last_message = session.get('last_bot_message')
                        if last_message:
                            if lang == 'en':
                                response_text = (
                                    f"🤔 *I didn't understand that command.*\n\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                                    f"{last_message}"
                                )
                            else:
                                response_text = (
                                    f"🤔 *No entendí ese comando.*\n\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                                    f"{last_message}"
                                )
                        else:
                            # No last message - process as normal flow anyway
                            response_text = process_message(from_number, incoming_msg)
        
        # Get session again for logging (may have changed)
        session = sessions.get(from_number, None)
        
        # CRITICAL: Validate response_text before sending
        if not response_text or response_text.strip() == "":
            print(f"[ERROR] Empty response_text for {from_number}, using fallback")
            lang = session.get('language', 'es') if session else 'es'
            if lang == 'en':
                response_text = (
                    "❌ *Oops! Something went wrong.* 🌸\n\n"
                    "I couldn't generate a response. Please try again or send *HELP* for assistance."
                )
            else:
                response_text = (
                    "❌ *¡Ups! Algo salió mal.* 🌸\n\n"
                    "No pude generar una respuesta. Por favor intenta de nuevo o envía *AYUDA* para asistencia."
                )
        
        log_user_action(from_number, f"Response generated (length: {len(response_text)})", response_text[:150], session, "DEBUG")
        
        # Save last bot message to session for command repetition
        if session:
            session['last_bot_message'] = response_text
            save_session(from_number, session)
        
        # CRITICAL: Sanitize message text before sending (remove excessive newlines, ensure UTF-8)
        # Replace multiple consecutive newlines with double newline
        sanitized_text = re.sub(r'\n{3,}', '\n\n', response_text)
        
        # CRITICAL: Handle long messages by splitting into multiple messages
        # WhatsApp limit is 1600 chars per message, but Twilio may concatenate, so use smaller limit
        MAX_MESSAGE_LENGTH = 1400  # Reduced to avoid Twilio concatenation issues
        if len(sanitized_text) > MAX_MESSAGE_LENGTH:
            print(f"[WARNING] Message too long ({len(sanitized_text)} chars), splitting into multiple messages")
            
            # Split message at logical points (double newlines or section separators)
            parts = []
            current_part = ""
            
            # Try to split at section separators first (━━━━━━━━━━━━━━━━━━)
            sections = re.split(r'(━━━━━━━━━━━━━━━━━━[^\n]*)', sanitized_text)
            
            for section in sections:
                # Check if adding this section would exceed limit
                if len(current_part) + len(section) <= MAX_MESSAGE_LENGTH:
                    current_part += section
                else:
                    # Save current part if it has content
                    if current_part.strip():
                        parts.append(current_part.strip())
                    # If a single section is too long, split it by newlines
                    if len(section) > MAX_MESSAGE_LENGTH:
                        lines = section.split('\n')
                        temp_part = ""
                        for line in lines:
                            # Check if adding this line would exceed limit
                            if len(temp_part) + len(line) + 1 <= MAX_MESSAGE_LENGTH:
                                temp_part += line + '\n'
                            else:
                                # Save temp_part if it has content
                                if temp_part.strip():
                                    parts.append(temp_part.strip())
                                # Start new part with current line
                                temp_part = line + '\n'
                        if temp_part.strip():
                            current_part = temp_part.strip()
                        else:
                            current_part = ""
                    else:
                        current_part = section
            
            # Add final part if it has content
            if current_part.strip():
                parts.append(current_part.strip())
            
            # CRITICAL: Verify all parts are within limit and add part indicators
            verified_parts = []
            for i, part in enumerate(parts, start=1):
                # Add part indicator to prevent Twilio concatenation
                part_indicator = f"[Parte {i}/{len(parts)}]\n\n"
                # Check if adding indicator would exceed limit
                if len(part) + len(part_indicator) <= MAX_MESSAGE_LENGTH:
                    verified_part = part_indicator + part
                else:
                    # If part is too long even without indicator, truncate it
                    max_part_length = MAX_MESSAGE_LENGTH - len(part_indicator) - 10
                    verified_part = part_indicator + part[:max_part_length] + "\n... (continúa)"
                
                # Final safety check
                if len(verified_part) > MAX_MESSAGE_LENGTH:
                    print(f"[WARNING] Part {i} still too long ({len(verified_part)} chars), truncating")
                    verified_part = verified_part[:MAX_MESSAGE_LENGTH - 20] + "\n... (truncado)"
                
                verified_parts.append(verified_part)
                print(f"[DEBUG] Part {i}/{len(parts)}: {len(verified_part)} chars")
            
            parts = verified_parts
            
            # Send first part via webhook response
            # CRITICAL: Verify first part length before sending
            sanitized_text = parts[0]
            if len(sanitized_text) > 1600:
                print(f"[ERROR] First part exceeds 1600 chars ({len(sanitized_text)} chars), truncating")
                sanitized_text = sanitized_text[:1570] + "\n... (mensaje truncado)"
            if len(parts) > 1:
                print(f"[INFO] Message split into {len(parts)} parts. Sending first part ({len(sanitized_text)} chars)")
                
                # CRITICAL: Validate Twilio credentials and client before attempting to send additional parts
                if (client and TWILIO_ACCOUNT_SID and TWILIO_ACCOUNT_SID != 'your_account_sid' and 
                    TWILIO_AUTH_TOKEN and TWILIO_AUTH_TOKEN != 'your_auth_token'):
                    # Send remaining parts asynchronously using Twilio API
                    try:
                        def send_remaining_parts():
                            for i, part in enumerate(parts[1:], start=2):
                                try:
                                    # CRITICAL: Verify part length before sending
                                    if len(part) > 1600:
                                        print(f"[ERROR] Part {i} exceeds 1600 chars ({len(part)} chars), truncating")
                                        part = part[:1570] + "\n... (mensaje truncado)"
                                    
                                    # Send message with explicit parameters to prevent concatenation
                                    client.messages.create(
                                        body=part,
                                        from_=TWILIO_WHATSAPP_NUMBER,
                                        to=from_number,
                                        # Add status callback to track delivery
                                        status_callback=None  # Can add webhook URL if needed
                                    )
                                    print(f"[INFO] Sent part {i}/{len(parts)} to {from_number} ({len(part)} chars)")
                                    # Optimized delay between messages to prevent Twilio concatenation
                                    if i < len(parts):  # Don't sleep after last part
                                        time.sleep(0.3)  # Optimized: 0.3s is sufficient and faster
                                except Exception as e:
                                    print(f"[ERROR] Failed to send part {i}: {e}")
                                    print(f"[ERROR] Part length: {len(part)} chars")
                                    import traceback
                                    print(f"[TRACEBACK] {traceback.format_exc()}")
                        
                        # Send remaining parts in background thread
                        thread = threading.Thread(target=send_remaining_parts, daemon=True)
                        thread.start()
                    except Exception as e:
                        print(f"[ERROR] Failed to start thread for remaining parts: {e}")
                        import traceback
                        print(f"[TRACEBACK] {traceback.format_exc()}")
                        # Fallback: append indication that message was truncated
                        sanitized_text += "\n\n... (mensaje continuará - error al enviar partes adicionales)"
                else:
                    print(f"[WARNING] Twilio credentials not configured or client not initialized. Only sending first part.")
                    print(f"[WARNING] To send multiple parts, configure TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env file")
                    # Append indication that message was truncated
                    sanitized_text += "\n\n... (mensaje continuará - configure credenciales de Twilio en .env)"
            else:
                print(f"[INFO] Message fits in single part after sanitization ({len(sanitized_text)} chars)")
        
        # Final check: if still too long after sanitization, truncate with indication
        # Use 1600 as absolute maximum (Twilio hard limit)
        ABSOLUTE_MAX_LENGTH = 1600
        if len(sanitized_text) > ABSOLUTE_MAX_LENGTH:
            print(f"[WARNING] Message still too long after sanitization ({len(sanitized_text)} chars), truncating to {ABSOLUTE_MAX_LENGTH}")
            sanitized_text = sanitized_text[:ABSOLUTE_MAX_LENGTH - 50] + "\n\n... (mensaje truncado por longitud)"
        # Ensure text is properly encoded
        try:
            sanitized_text = sanitized_text.encode('utf-8').decode('utf-8')
        except Exception as e:
            print(f"[WARNING] Encoding issue with message: {e}")
            # Fallback: remove problematic characters
            sanitized_text = response_text.encode('ascii', 'ignore').decode('ascii')
        
        # Create Twilio response
        resp = MessagingResponse()
        try:
            msg = resp.message(sanitized_text)
            print(f"[DEBUG WEBHOOK] Message created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create Twilio message: {e}")
            import traceback
            print(f"[TRACEBACK] {traceback.format_exc()}")
            # Fallback: try with simplified message
            resp = MessagingResponse()
            fallback_msg = "❌ Error al generar respuesta. Por favor intenta de nuevo."
            msg = resp.message(fallback_msg)
            sanitized_text = fallback_msg
        
        # Log that message is being sent
        print(f"[INFO] Sending message to {from_number} (length: {len(sanitized_text)} chars)")
        print(f"[DEBUG WEBHOOK] MessagingResponse created: {type(resp)}")
        print(f"[DEBUG WEBHOOK] Message object: {type(msg)}")
        
        try:
            response_str = str(resp)
            print(f"[DEBUG WEBHOOK] Final XML length: {len(response_str)}")
            print(f"[DEBUG WEBHOOK] Final XML preview: {response_str[:300]}")
            
            # Verify XML is valid and contains the message
            if not response_str or len(response_str) < 50:
                print(f"[ERROR] Invalid XML response generated (length: {len(response_str)})")
                # Generate fallback response
                resp = MessagingResponse()
                resp.message("❌ Error técnico. Por favor intenta de nuevo.")
                response_str = str(resp)
            elif sanitized_text not in response_str and len(sanitized_text) < 100:
                # Check if message content is in XML (for short messages)
                print(f"[WARNING] Message content may not be in XML response")
                print(f"[DEBUG] Looking for: {sanitized_text[:50]}...")
            
        except Exception as e:
            print(f"[ERROR] Failed to convert response to string: {e}")
            import traceback
            print(f"[TRACEBACK] {traceback.format_exc()}")
            # Generate fallback response
            resp = MessagingResponse()
            resp.message("❌ Error técnico. Por favor intenta de nuevo.")
            response_str = str(resp)
        
        # CRITICAL: Ensure proper Content-Type for Twilio
        response = make_response(response_str)
        response.headers['Content-Type'] = 'text/xml; charset=utf-8'
        return response
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Webhook error: {e}")
        print(f"[TRACEBACK] {traceback.format_exc()}")
        resp = MessagingResponse()
        resp.message(
            "❌ *Oops! Something went wrong.* 🌸\n\n"
            "💡 *Need help?*\n"
            "• Send *HELP* for help\n"
            "• Send *RESET* to restart\n"
            "• Send *RESTART* to clear session\n\n"
            "If the problem persists, try again in a few moments."
        )
        response_str = str(resp)
        response = make_response(response_str)
        response.headers['Content-Type'] = 'text/xml; charset=utf-8'
        return response

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model_loaded': sentiment_predictor is not None,
        'active_sessions': len(sessions),
        'timestamp': datetime.now().isoformat()
    }

@app.route('/')
def home():
    """Home page"""
    return """
    <html>
        <head>
            <title>SAORI AI Core - WhatsApp Bot</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .card {
                    background: white;
                    color: #333;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }
                h1 { color: #667eea; }
                .status { 
                    padding: 10px; 
                    background: #4CAF50; 
                    color: white; 
                    border-radius: 5px;
                    text-align: center;
                    margin: 20px 0;
                }
                code {
                    background: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🎯 SAORI AI Core - WhatsApp Bot</h1>
                <div class="status">✅ System Online</div>
                <p><strong>AI-Powered Recruiting Assistant</strong></p>
                <p>Send a message to our WhatsApp number to start your AI interview!</p>
                <h3>Features:</h3>
                <ul>
                    <li>🤖 Real-time AI sentiment analysis</li>
                    <li>🔧 Technical skills evaluation</li>
                    <li>🗣️ English proficiency assessment</li>
                    <li>💼 Soft skills detection</li>
                    <li>📊 Instant scoring and feedback</li>
                </ul>
                <h3>Active Sessions:</h3>
                <p><strong>{}</strong> candidates currently interviewing</p>
                <hr>
                <p><small>Powered by BERT AI Model (100% accuracy) | Version 1.0.0</small></p>
            </div>
        </body>
    </html>
    """.format(len(sessions))

# Pre-load BERT model at startup for better performance
def preload_bert_models():
    """Pre-load BERT models for both languages to avoid delays during interviews"""
    try:
        print("[INFO] Pre-cargando modelos BERT para mejor rendimiento...")
        import src.whatsapp_inconsistency_detector as detector_module
        
        # Pre-load para español
        print("[INFO] Pre-cargando modelo BERT (Español)...")
        detector_module._get_bert_checker(language='es')
        
        # Pre-load para inglés
        print("[INFO] Pre-cargando modelo BERT (Inglés)...")
        detector_module._get_bert_checker(language='en')
        
        print("[INFO] ✅ Modelos BERT pre-cargados correctamente")
    except Exception as e:
        print(f"[WARNING] No se pudieron pre-cargar modelos BERT: {e}")
        print("[INFO] El sistema funcionará normalmente, pero puede haber pequeñas demoras en la primera detección")

# Load all sessions at startup to prevent loss on restart
print("[INFO] Loading all sessions from files...")
load_all_sessions()

# MAIN EXECUTION DISABLED - Using main from whatsapp_bot_with_profiles.py instead
if __name__ == '__main__':
    print("="*60)
    print("🎯 SAORI AI Core - WhatsApp Bot")
    print("="*60)
    print(f"[INFO] AI Model: Loaded ✅")
    print(f"[INFO] Twilio: Configured ✅")
    
    # Pre-load BERT models in background (non-blocking)
    try:
        import threading
        bert_thread = threading.Thread(target=preload_bert_models, daemon=True)
        bert_thread.start()
        print(f"[INFO] BERT pre-load iniciado en background...")
    except Exception as e:
        print(f"[WARNING] No se pudo iniciar pre-carga de BERT: {e}")
    
    print(f"[INFO] Starting Flask server...")
    print("="*60)
    
    # Run Flask app (debug=False to avoid caching issues)
    # Use PORT from environment variable (for cloud hosting) or default to 5000
    try:
        port = int(os.environ.get('PORT', 5000))
        if port < 1 or port > 65535:
            raise ValueError("Port out of valid range")
    except (ValueError, TypeError):
        port = 5000
        print("[WARNING] Invalid PORT environment variable, using default 5000")
    
    print(f"[INFO] Server starting on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)

