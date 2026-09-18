# Keywords for rule-based classification
UNAVAILABILITY_KWS = [
    "cannot work", "can't work", "cannot make it", "unable to work",
    "not available", "won't be able to work", "do not schedule",
    "don't schedule", "cannot come", "won't make it", "no availability"
]

REQUEST_KWS = [
    "we need", "need ", "requires", "shift open", "opening available",
    "role open", "vacant shift", "any takers"
]

AVAIL_KWS = [
    "available", "free", "can work", "can only work", "is available", "is free"
]


def classify_intent(line: str, parent: str, classifier) -> str:
    """
    Classify intent using rule-based + ML hybrid approach
    
    Args:
        line: Current text fragment
        parent: Original full line (for context)
        classifier: Trained ML classifier
        
    Returns:
        Intent label: AVAILABILITY, UNAVAILABILITY, SHIFT_REQUEST, or PREFERENCE
    """
    line_l = line.lower()
    parent_l = parent.lower()

    # Rule 1: Check for unavailability keywords
    if any(kw in parent_l for kw in UNAVAILABILITY_KWS) or any(kw in line_l for kw in UNAVAILABILITY_KWS):
        return "UNAVAILABILITY"

    # Rule 2: Check for shift request keywords
    if any(kw in parent_l for kw in REQUEST_KWS) or any(kw in line_l for kw in REQUEST_KWS):
        return "SHIFT_REQUEST"

    # Rule 3: Check for availability keywords
    if any(kw in line_l for kw in AVAIL_KWS):
        return "AVAILABILITY"

    # Rule 4: If parent has availability and child doesn't mention preference
    if (any(kw in parent_l for kw in AVAIL_KWS) and 
        not ("prefer" in line_l or "prefers" in line_l)):
        return "AVAILABILITY"

    # Fallback to ML classifier
    return classifier.predict([line])[0]