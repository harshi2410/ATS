"""
Candidate Reporting Module.
Exports ranking summaries to CSV, JSON, and rich formatted text reports.
"""

import csv
import json
from typing import List, Dict, Any, Optional
import pandas as pd


def generate_ranking_dataframe(ranked_candidates: List[Dict[str, Any]]) -> pd.DataFrame:
    """Creates a clean pandas DataFrame summarizing all ranked candidates."""
    rows = []
    for c in ranked_candidates:
        rows.append({
            "Rank": c["rank"],
            "Candidate Name": c["name"],
            "Total Match (%)": c["total_score"],
            "Skill Match (%)": c["skill_match_score"],
            "TF-IDF Similarity (%)": c["tfidf_score"],
            "Matched Skills": ", ".join(c["matched_skills"]) if c["matched_skills"] else "None",
            "Missing Skills": ", ".join(c["missing_skills"]) if c["missing_skills"] else "None",
            "Degrees": ", ".join(c["degrees"]) if c["degrees"] else "N/A",
            "Email": c["contact"].get("email") or "N/A",
            "Phone": c["contact"].get("phone") or "N/A",
            "Filename": c["filename"]
        })
    return pd.DataFrame(rows)


def export_ranking_to_csv(ranked_candidates: List[Dict[str, Any]], filepath: str) -> str:
    """Exports candidate ranking list to a CSV file."""
    df = generate_ranking_dataframe(ranked_candidates)
    df.to_csv(filepath, index=False, encoding="utf-8")
    return filepath


def export_ranking_to_json(ranked_candidates: List[Dict[str, Any]], filepath: str) -> str:
    """Exports full candidate data to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ranked_candidates, f, indent=2)
    return filepath


def generate_text_summary(candidate: Dict[str, Any]) -> str:
    """Generates a comprehensive readable summary card for a single candidate."""
    contact = candidate.get("contact", {})
    email = contact.get("email") or "N/A"
    phone = contact.get("phone") or "N/A"
    degrees = ", ".join(candidate.get("degrees", [])) or "Not explicitly listed"

    matched = ", ".join(candidate.get("matched_skills", [])) or "None"
    missing = ", ".join(candidate.get("missing_skills", [])) or "None"
    extra = ", ".join(candidate.get("additional_skills", [])[:8]) or "None"
    keywords = ", ".join(candidate.get("top_matching_keywords", [])) or "N/A"

    report = f"""================================================================================
CANDIDATE RANK #{candidate.get('rank')}: {candidate.get('name')}
File: {candidate.get('filename')}
================================================================================
Match Scores:
  • Overall ATS Match Score:   {candidate.get('total_score')}%
  • Direct Skill Overlap:      {candidate.get('skill_match_score')}%
  • Content TF-IDF Similarity: {candidate.get('tfidf_score')}%
  • Section Completeness:      {candidate.get('completeness_score')}%

Contact & Academic Info:
  • Email:    {email}
  • Phone:    {phone}
  • Degrees:  {degrees}

Skill Breakdown:
  • Matched Required Skills:  {matched}
  • Missing Required Skills:  {missing}
  • Other Notable Skills:     {extra}
  • Shared Keywords:          {keywords}

Sections Identified:
  • Experience:    {'Yes' if candidate.get('sections', {}).get('experience') else 'No'}
  • Education:     {'Yes' if candidate.get('sections', {}).get('education') else 'No'}
  • Skills:        {'Yes' if candidate.get('sections', {}).get('skills') else 'No'}
================================================================================
"""
    return report
