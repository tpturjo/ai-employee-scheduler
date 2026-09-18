import streamlit as st
import pandas as pd
from models.nlp_model import load_nlp_resources
from models.classifier import classify_intent
from utils.parsers import preprocess_lines, extract_details
from scheduler.engine import generate_roster
from scheduler.analysis import calculate_fairness_score, generate_llm_summary

#PAGE CONFIG
st.set_page_config(page_title="SmartScheduler", page_icon="📅", layout="wide")

#LOAD RESOURCES
nlp, matcher, classifier = load_nlp_resources()

#UI 
st.title("SmartScheduler: NLP-Powered Employee Scheduling")


# Sidebar
with st.sidebar:
    st.header("### Project BY")
    st.markdown("## Tridib Paul Turjo")
    st.markdown("## Md Golam Sharier")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses:
    - **spaCy** for NLP
    - **Naive Bayes** for classification
    - **Mistral-7B** (FREE) for summaries
    """)
    
    st.markdown("---")
    st.markdown("### Key Features:")
    st.markdown("""
    - Parses informal text
    - Extracts entities (names, days, times, roles)
    - Classifies intent (AVAILABILITY, UNAVAILABILITY, SHIFT_REQUEST, PREFERENCE)
    - Assigns shifts while avoiding conflicts
    - Generates AI explanations and recommendations""")


# Default example text
default_text = """Alice can only work Monday 9-5 as a cashier and Wednesday 1-9 on the floor.
Bob is available Tuesday 3-11 for stock and Friday 9-5 as a cashier.
Claire prefers mornings and is free on Friday 9-1 as a supervisor.
Dan cannot work on Tuesday because of a doctor's appointment.
Erin is available all day Saturday for any role.
Frank is available Monday 9-5 and Tuesday 9-5 but prefers evenings.
We need one cashier on Monday 9-5.
We need two people in stock on Tuesday 3-11.
We need a supervisor on Friday 9-1.
We need a cashier on Friday 9-5.
We need one more person for general help on Saturday 12-8.
Sam has an exam on Friday and cannot work that day."""

raw_text = st.text_area("Constraints / Staff Messages:", value=default_text, height=300)

if st.button("Generate Schedule"):
    final_shifts, conflict_log, processed_items = generate_roster(raw_text, nlp, matcher, classifier)
    
    # Build employee shift assignments
    employee_shifts = {}
    for s in final_shifts:
        if s["Assigned"]:
            name = s["Assigned"]["Name"]
            day = s["Day"] if s["Day"] else "TBD"
            time = s["Time"]
            role = s["Role"]
            if name not in employee_shifts:
                employee_shifts[name] = []
            employee_shifts[name].append(f"{day} {time} ({role})")

    # Display intent classification
    st.subheader("1. AI Intent Classification")
    with st.expander("Click to view how each fragment was classified"):
        for item in processed_items:
            line = item["text"]
            parent = item["parent"]
            intent = classify_intent(line, parent, classifier)
            color = (
                "green" if intent == "AVAILABILITY"
                else "red" if intent == "UNAVAILABILITY"
                else "blue" if intent == "SHIFT_REQUEST"
                else "orange"
            )
            st.markdown(f":{color}[**{intent}**] → `{line}`")

    # Display conflicts
    st.subheader("2. Conflicts Detected")
    if conflict_log:
        for c in conflict_log:
            st.error(c)
    else:
        st.success("No conflicts detected 🎉")

    # Display schedule
    st.subheader("3. Final Schedule")
    schedule_data = []
    for s in final_shifts:
        assignee = s["Assigned"]["Name"] if s["Assigned"] else "UNFILLED"
        
        pref = ""
        if s["Assigned"] and s["Assigned"].get("Preference"):
            emp_pref = s["Assigned"]["Preference"]
            if s["Day"] and str(s["Day"]).lower() in emp_pref.lower():
                pref = emp_pref
            elif s["Time"] != "Any" and str(s["Time"]) in emp_pref:
                pref = emp_pref
            elif any(word in emp_pref.lower() for word in ["prefer", "morning", "evening", "night"]):
                if not any(day in emp_pref.lower() for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
                    pref = emp_pref

        status = "✅ Scheduled" if s["Assigned"] else "❌ Unfilled"
        if "Inferred" in s["Source"]:
            status = "ℹ️ Added (Availability)"

        schedule_data.append({
            "Day": s["Day"],
            "Time": s["Time"],
            "Role": s["Role"],
            "Employee": assignee,
            "Preference Note": pref,
            "Status": status,
        })

    df = pd.DataFrame(schedule_data)
    st.table(df)

    # Display summary
    st.subheader("4. Summary & Interpretation")
    filled = sum(1 for s in final_shifts if s["Assigned"])
    total = len(final_shifts)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fill Rate", f"{filled}/{total} shifts")
    with col2:
        st.metric("Conflicts", len(conflict_log))
    with col3:
        fairness_score = calculate_fairness_score(final_shifts)
        st.metric("Fairness Score", f"{fairness_score}/100")
    
    st.markdown("### 🤖 AI-Generated Schedule Analysis")
    llm_summary = generate_llm_summary(final_shifts, conflict_log, employee_shifts)
    st.markdown(llm_summary)
    
    st.write("**Detailed Employee Assignments:**")
    for name, shifts_list in employee_shifts.items():
        if len(shifts_list) > 2:
            st.error(f"🚨 **{name}**: {len(shifts_list)} shifts (OVERWORKED) - {', '.join(shifts_list)}")
        elif len(shifts_list) == 2:
            st.info(f"📋 **{name}**: {len(shifts_list)} shifts - {', '.join(shifts_list)}")
        else:
            st.success(f"✓ **{name}**: {', '.join(shifts_list)}")