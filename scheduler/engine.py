import re
from typing import List, Dict, Any, Tuple, Optional
from models.classifier import classify_intent
from utils.parsers import preprocess_lines, extract_details


def generate_roster(raw_text: str, nlp, matcher, classifier) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, str]]]:
    """
    Main scheduling algorithm
    
    Process:
    1. Parse and classify input lines
    2. Extract employee availability and shift requests
    3. Match employees to shifts using constraint-based scoring
    4. Create inferred shifts for unassigned employees
    5. Detect conflicts
    
    Returns:
        (shifts, conflicts, processed_items)
    """
    items = preprocess_lines(raw_text, nlp)

    employees = []
    shifts = []
    conflicts = []
    conflict_people = set()

    last_person_seen = None

    #Parse all inputs and classify
    for item in items:
        line = item["text"]
        parent = item["parent"]
        inherited_person = item.get("inherited_person")

        intent = classify_intent(line, parent, classifier)
        details = extract_details(line, parent, intent, nlp, matcher, last_person_seen, inherited_person)

        if details["Name"]:
            last_person_seen = details["Name"]

        line_lower = line.lower()

        if intent == "UNAVAILABILITY":
            name = details["Name"] or last_person_seen or "Unknown"
            conflict_key = f"{name}:{parent}"
            if conflict_key not in conflict_people:
                conflicts.append(f"❌ **{name}** is unavailable ({parent})")
                conflict_people.add(conflict_key)

        elif intent == "AVAILABILITY":
            # Check if this is actually a preference without day/time
            if ("prefer" in line_lower or "prefers" in line_lower) and details["Day"] is None and details["Time"] == "Any":
                target_name = details["Name"] or last_person_seen
                if target_name:
                    for emp in employees:
                        if emp["Name"] == target_name:
                            emp["Preference"] = line
                continue

            employees.append({
                "Name": details["Name"],
                "Day": details["Day"],
                "Time": details["Time"],
                "Role": details["Role"],
                "Preference": None,
                "Is_Assigned": False,
            })

            if "prefer" in line_lower or "prefers" in line_lower:
                if employees:
                    employees[-1]["Preference"] = line

        elif intent == "PREFERENCE":
            target_name = details["Name"] or last_person_seen
            if target_name:
                for emp in employees:
                    if emp["Name"] == target_name:
                        emp["Preference"] = line
                        break

        elif intent == "SHIFT_REQUEST":
            # Extract number of positions needed
            clean_line = line_lower
            clean_line = re.sub(r"\d+\s*[-:]\s*\d+", " ", clean_line)
            clean_line = re.sub(r"\d+\s+to\s+\d+", " ", clean_line)

            count = 1
            if "two" in clean_line or re.search(r"\b2\b", clean_line):
                count = 2
            elif "three" in clean_line or re.search(r"\b3\b", clean_line):
                count = 3

            for _ in range(count):
                shifts.append({
                    "Day": details["Day"],
                    "Time": details["Time"],
                    "Role": details["Role"],
                    "Assigned": None,
                    "Source": "Explicit Request",
                })

    # Match employees to shifts using constraint-based scoring
    for shift in shifts:
        best_match = None
        best_score = 0

        for emp in employees:
            if emp["Is_Assigned"]:
                continue

            score = 0

            # Day matching (highest priority)
            if shift["Day"] and emp["Day"]:
                s_day = str(shift["Day"]).lower()
                e_day = str(emp["Day"]).lower()
                if s_day == e_day or s_day in e_day or e_day in s_day:
                    score += 5
                else:
                    continue  # Hard constraint: day must match
            elif shift["Day"] is None and emp["Day"]:
                score += 1
            elif shift["Day"] and emp["Day"] is None:
                score += 1

            # Role matching
            if shift["Role"] == emp["Role"]:
                score += 5
            elif shift["Role"] == "General":
                score += 1
            elif emp["Role"] == "General":
                score += 2

            # Time matching
            if shift["Time"] == emp["Time"]:
                score += 3
            elif shift["Time"] == "Any" or emp["Time"] == "Any":
                score += 1

            if score > best_score:
                best_score = score
                best_match = emp

        # Assign if score meets threshold
        if best_match and best_score >= 5:
            shift["Assigned"] = best_match
            best_match["Is_Assigned"] = True

    #Create inferred shifts for unassigned employees
    for emp in employees:
        if not emp["Is_Assigned"]:
            emp_day = emp["Day"]
            emp_time = emp["Time"]
            emp_name = emp["Name"]
            
            # Check if already scheduled for this exact slot
            employee_already_scheduled = False
            for shift in shifts:
                if (shift["Assigned"] and 
                    shift["Assigned"]["Name"] == emp_name and
                    shift["Day"] == emp_day and 
                    shift["Time"] == emp_time):
                    employee_already_scheduled = True
                    break
            
            if not employee_already_scheduled:
                shifts.append({
                    "Day": emp["Day"] or "TBD",
                    "Time": emp["Time"],
                    "Role": emp["Role"],
                    "Assigned": emp,
                    "Source": "Inferred from Availability",
                })
                emp["Is_Assigned"] = True

    return shifts, conflicts, items