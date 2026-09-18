import re
from typing import List, Dict, Optional, Any


def normalize_role(role_text: str) -> str:
    """Normalize role text to standard categories"""
    if not role_text:
        return "General"
    
    role_map = {
        "stock": "Stock", "restock": "Stock", "inventory": "Stock",
        "cashier": "Cashier", "register": "Cashier",
        "supervisor": "Supervisor", "manager": "Supervisor",
        "floor": "General", "help": "General", "general": "General",
    }
    
    lower = role_text.lower()
    for key, val in role_map.items():
        if key in lower:
            return val
    return "General"


def preprocess_lines(raw_text: str, nlp) -> List[Dict[str, str]]:
    """
    Split input text into processable fragments
    
    Handles:
    - Newline separation
    - "and" / ";" separation
    - Person name inheritance
    """
    processed = []
    base_lines = raw_text.split("\n")

    for line in base_lines:
        if not line.strip():
            continue

        # Split by " and " or "; "
        parts = re.split(r" and |; ", line)
        
        # Extract person from full line for inheritance
        doc = nlp(line)
        line_person = None
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                line_person = ent.text
                break
        
        for part in parts:
            part = part.strip()
            if part:
                # Check if part has its own person
                part_doc = nlp(part)
                part_has_person = any(ent.label_ == "PERSON" for ent in part_doc.ents)
                
                processed.append({
                    "text": part,
                    "parent": line,
                    "inherited_person": line_person if not part_has_person else None
                })

    return processed


def extract_details(text: str, parent: str, intent: str, nlp, matcher,
                    last_person: Optional[str] = None, 
                    inherited_person: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract structured data from text
    
    Extracts:
    - Name (PERSON entity)
    - Day (DATE entity or weekday regex)
    - Time (pattern matching)
    - Role (keyword normalization)
    """
    doc = nlp(text)
    parent_doc = nlp(parent)
    
    data = {"Name": None, "Day": None, "Time": "Any", "Role": "General"}

    # Extract person name (priority: current text → inherited → parent → last seen)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            data["Name"] = ent.text
            break
    
    if data["Name"] is None and inherited_person:
        data["Name"] = inherited_person
    
    if data["Name"] is None:
        for ent in parent_doc.ents:
            if ent.label_ == "PERSON":
                data["Name"] = ent.text
                break
    
    if data["Name"] is None and last_person and intent in ("AVAILABILITY", "UNAVAILABILITY", "PREFERENCE"):
        data["Name"] = last_person

    # Extract time using Matcher
    matches = matcher(doc)
    time_spans = []
    for _, start, end in matches:
        data["Time"] = doc[start:end].text
        time_spans.append((start, end))
        break

    # Extract day (weekday pattern or DATE entity)
    days_pattern = r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
    
    day_match = re.search(days_pattern, text, re.IGNORECASE)
    if day_match:
        data["Day"] = day_match.group(1)
    else:
        day_match = re.search(days_pattern, parent, re.IGNORECASE)
        if day_match:
            data["Day"] = day_match.group(1)
    
    # Fallback to DATE entities (avoid time overlap)
    if data["Day"] is None:
        time_token_indices = set()
        for start, end in time_spans:
            for i in range(start, end):
                time_token_indices.add(i)
        
        for ent in doc.ents:
            if ent.label_ == "DATE":
                overlaps = any(token.i in time_token_indices for token in ent)
                if not overlaps:
                    data["Day"] = ent.text
                    break
        
        if data["Day"] is None:
            parent_matches = matcher(parent_doc)
            parent_time_indices = set()
            for _, start, end in parent_matches:
                for i in range(start, end):
                    parent_time_indices.add(i)
            
            for ent in parent_doc.ents:
                if ent.label_ == "DATE":
                    overlaps = any(token.i in parent_time_indices for token in ent)
                    if not overlaps:
                        data["Day"] = ent.text
                        break

    # Extract role
    data["Role"] = normalize_role(text.lower())

    return data