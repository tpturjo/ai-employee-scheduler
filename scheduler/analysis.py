import json
import streamlit as st
from typing import List, Dict, Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError


def calculate_fairness_score(shifts: List[Dict[str, Any]]) -> int:
    """
    Calculate fairness score based on workload distribution

    Uses variance to measure how evenly shifts are distributed.
    Lower variance = higher fairness.
    """
    employee_counts = {}
    for shift in shifts:
        if shift["Assigned"]:
            name = shift["Assigned"]["Name"]
            employee_counts[name] = employee_counts.get(name, 0) + 1

    if not employee_counts:
        return 100

    counts = list(employee_counts.values())
    avg = sum(counts) / len(counts)
    variance = sum((x - avg) ** 2 for x in counts) / len(counts)

    fairness = max(0, 100 - (variance * 25))
    return int(fairness)


def generate_llm_summary(
    shifts: List[Dict[str, Any]],
    conflicts: List[str],
    employee_assignments: Dict,
    hf_token: str = None,
) -> str:
    """
    Generate natural language summary using Hugging Face **Serverless Inference API**
    via huggingface_hub.InferenceClient (chat_completion).

    If anything fails (quota, 404, model unsupported, etc.), we fall back
    to the enhanced template-based summary.
    """

    if hf_token is None:
        hf_token = st.secrets.get("HF_TOKEN", None)
        
    # Prepare metrics
    filled = sum(1 for s in shifts if s["Assigned"])
    total = len(shifts)
    unfilled = total - filled
    explicit = sum(
        1 for s in shifts if "Explicit" in s.get("Source", "") and s["Assigned"]
    )
    inferred = sum(1 for s in shifts if "Inferred" in s.get("Source", ""))
    fairness_score = calculate_fairness_score(shifts)

    context = {
        "total_shifts": total,
        "filled_shifts": filled,
        "unfilled_shifts": unfilled,
        "explicit_matches": explicit,
        "inferred_shifts": inferred,
        "conflicts_count": len(conflicts),
        "fairness_score": fairness_score,
        "employee_workload": {
            name: len(shift_list) for name, shift_list in employee_assignments.items()
        },
    }

    # If no token or user doesn't want online calls, just use template.
    if not hf_token:
        return generate_enhanced_summary(context, employee_assignments, conflicts)

    # Build a compact "report" as the chat message
    prompt = f"""
You are an expert workforce scheduler. Analyze this employee schedule professionally and concisely.

Data:
- Total shifts: {total}, Filled: {filled}, Unfilled: {unfilled}
- Explicitly matched preference shifts: {explicit}
- Inferred shifts (from availability): {inferred}
- Number of conflicts respected: {len(conflicts)}
- Fairness score (0–100): {fairness_score}

Employee workload by name (number of assigned shifts):
{json.dumps(context["employee_workload"], indent=2)}

Conflicts:
{json.dumps(conflicts, indent=2)}

Please respond in markdown with:
1. A brief overall assessment (1–2 sentences)
2. 3–6 key points (coverage, fairness, conflicts, obvious issues)
3. 2–4 practical recommendations to improve the schedule if needed.
"""

    try:
        # InferenceClient will use HF Inference (serverless) by default.
        # We explicitly choose a model that HF shows in their docs as usable.
        client = InferenceClient(
            provider = "auto",
            token=hf_token,  # or api_key=hf_token
            timeout=30,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a concise, professional workforce scheduling analyst.",
            },
            {"role": "user", "content": prompt},
        ]

        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=messages,
            max_tokens=700,
            temperature=0.6,
            top_p=0.95,
        )

        st.write("HF response received:", completion)
        
        if not completion.choices:
            st.info("ℹ️ HF returned no response. Using template analysis.")
            return generate_enhanced_summary(
                context, employee_assignments, conflicts
            )
            

        # According to HF docs, this is the structure:
        # ChatCompletionOutput -> choices[0].message.content
        llm_text = completion.choices[0].message.content if completion.choices else ""

        if llm_text and len(llm_text.strip()) > 50:
            return f"**🤖 AI-Generated Analysis (Hugging Face)**\n\n{llm_text}"

        # If it's suspiciously short / empty, fall back
        st.info("ℹ️ HF LLM returned empty/short content. Using template analysis.")
        return generate_enhanced_summary(context, employee_assignments, conflicts)

    except HfHubHTTPError as e:
        # Typical HF serverless errors: 401, 403, 404, 429, 5xx…
        st.warning(f"⚠️ HF LLM HTTP error: {e}. Using template analysis.")
        return generate_enhanced_summary(context, employee_assignments, conflicts)
    except Exception as e:
        # Any other weird issue -> template
        st.warning(f"⚠️ LLM error: {e}. Using template analysis.")
        return generate_enhanced_summary(context, employee_assignments, conflicts)


def generate_enhanced_summary(
    context: Dict, employee_assignments: Dict, conflicts: List[str]
) -> str:
    """Enhanced template-based summary that mimics AI analysis."""
    filled = context["filled_shifts"]
    total = context["total_shifts"]
    unfilled = context["unfilled_shifts"]
    fairness = context["fairness_score"]

    # Overall Assessment
    if unfilled == 0:
        assessment = (
            f"**📊 Schedule Analysis (Enhanced Template)**\n\n"
            f"**Overall Assessment:** Excellent scheduling outcome! "
            f"All {total} required shifts have been successfully filled with available staff. "
        )
    else:
        fill_rate = (filled / total * 100) if total > 0 else 0
        if fill_rate >= 80:
            assessment = (
                f"**📊 Schedule Analysis (Enhanced Template)**\n\n"
                f"**Overall Assessment:** Good progress with {filled} out of {total} shifts "
                f"filled ({fill_rate:.0f}% coverage). "
            )
        else:
            assessment = (
                f"**📊 Schedule Analysis (Enhanced Template)**\n\n"
                f"**Overall Assessment:** Significant staffing gaps remain with only {filled} "
                f"out of {total} shifts covered ({fill_rate:.0f}%). "
            )

    # Fairness analysis
    if fairness >= 90:
        assessment += "Workload distribution is exceptionally balanced across the team. "
    elif fairness >= 70:
        assessment += "Workload distribution shows some minor imbalances but remains acceptable. "
    else:
        assessment += (
            "Notable workload imbalances detected that may lead to employee dissatisfaction. "
        )

    # Conflict analysis
    if context["conflicts_count"] > 0:
        assessment += (
            f"However, {context['conflicts_count']} scheduling conflict(s) were identified "
            f"and respected."
        )
    else:
        assessment += "No scheduling conflicts detected."

    # Key Highlights
    highlights = "\n\n**Key Highlights:**\n"

    if context["explicit_matches"] > 0:
        highlights += (
            f"• ✅ Successfully matched {context['explicit_matches']} explicit shift requests\n"
        )

    if context["inferred_shifts"] > 0:
        highlights += (
            f"• 📋 Created {context['inferred_shifts']} additional shifts "
            f"based on employee availability\n"
        )

    if employee_assignments:
        workloads = list(context["employee_workload"].values())
        if workloads:
            max_shifts = max(workloads)
            min_shifts = min(workloads)
            avg_shifts = sum(workloads) / len(workloads)

            if max_shifts > avg_shifts * 1.5:
                overworked = [
                    name
                    for name, count in context["employee_workload"].items()
                    if count == max_shifts
                ]
                highlights += (
                    "• ⚠️ Workload concern: "
                    f"{', '.join(overworked)} assigned {max_shifts} shifts "
                    f"(above average of {avg_shifts:.1f})\n"
                )

            if max_shifts == min_shifts:
                highlights += (
                    f"• ⭐ Perfect balance: All employees assigned exactly "
                    f"{max_shifts} shift(s)\n"
                )

    if unfilled > 0:
        highlights += (
            f"• ❌ Critical gap: {unfilled} unfilled position(s) "
            f"requiring immediate attention\n"
        )

    # Recommendations
    recommendations = "\n\n**Recommendations:**\n"

    if unfilled > 0:
        if unfilled <= 2:
            recommendations += (
                f"• 🎯 **Priority Action:** Recruit {unfilled} additional team member(s) "
                f"or offer overtime incentives\n"
            )
        else:
            recommendations += (
                f"• 🚨 **Urgent:** Significant understaffing ({unfilled} positions). "
                f"Consider temporary staffing agency or shift consolidation\n"
            )

    if fairness < 70:
        recommendations += (
            "• ⚖️ **Fairness Issue:** Review and redistribute shifts to improve "
            "workload balance\n"
        )

    if context["conflicts_count"] > 2:
        recommendations += (
            "• 📅 **Availability Concern:** High number of conflicts suggests need for "
            "more flexible scheduling or a larger staff pool\n"
        )

    if unfilled == 0 and fairness >= 90:
        recommendations += (
            "• 🎉 **Excellent Work:** This schedule optimally balances coverage and "
            "fairness. No changes needed!\n"
        )
    else:
        recommendations += (
            "• 💡 **Ongoing:** Continue monitoring employee satisfaction and adjust "
            "preferences as needed\n"
        )

    return assessment + highlights + recommendations
