"""
Information Extractor Module.
Extracts:
1. Contact Details (Candidate Name, Email, Phone, GitHub/LinkedIn)
2. Resume Sections (Skills, Experience, Education, Projects, Summary)
3. Domain Skills using exact phrase and n-gram keyword matching.
"""

import re
from typing import Dict, List, Set, Any, Optional
from src.config import (
    DOMAIN_SKILLS, ALL_SKILLS, SECTION_HEADERS,
    EMAIL_PATTERN, PHONE_PATTERN, LINKEDIN_PATTERN, GITHUB_PATTERN, DEGREE_PATTERNS
)
from src.preprocessor import protect_tech_tokens


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """Extracts email, phone, LinkedIn, and GitHub links from resume text."""
    contact: Dict[str, Optional[str]] = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None
    }

    email_match = re.search(EMAIL_PATTERN, text)
    if email_match:
        contact["email"] = email_match.group(0).strip()

    phone_match = re.search(PHONE_PATTERN, text)
    if phone_match:
        contact["phone"] = phone_match.group(0).strip()

    linkedin_match = re.search(LINKEDIN_PATTERN, text, re.IGNORECASE)
    if linkedin_match:
        contact["linkedin"] = linkedin_match.group(0).strip()

    github_match = re.search(GITHUB_PATTERN, text, re.IGNORECASE)
    if github_match:
        contact["github"] = github_match.group(0).strip()

    return contact


def extract_candidate_name(text: str, filename: Optional[str] = None) -> str:
    """
    Infers candidate name from the top header lines or filename.
    Filters out emails, phones, URLs, and section titles.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Check top 6 lines
    for line in lines[:6]:
        # Skip if contains email, URL, phone, or colons
        if "@" in line or "http" in line or "www." in line or ":" in line:
            continue
        # Check if line looks like a header or label
        lower_line = line.lower()
        if any(lower_line.startswith(h) for h in ["resume", "curriculum", "cv", "page", "phone", "email"]):
            continue
        # Names are typically 2 to 4 capitalized words, 3 to 40 characters
        words = line.split()
        if 2 <= len(words) <= 4 and len(line) <= 45:
            # Check if largely alphabetical
            clean_line = re.sub(r"[^a-zA-Z\s.-]", "", line).strip()
            if len(clean_line) >= 4 and not any(char.isdigit() for char in line):
                return clean_line.title()

    # Fallback to filename if available
    if filename:
        clean_name = re.sub(r"[._-](pdf|txt|docx|resume|cv)\b", "", filename, flags=re.IGNORECASE)
        clean_name = re.sub(r"[_\-]+", " ", clean_name).strip()
        parts = clean_name.split()
        if parts:
            return " ".join(p.capitalize() for p in parts[:3])

    return "Candidate"


def extract_sections(text: str) -> Dict[str, str]:
    """
    Segments resume text into distinct sections:
    - experience
    - education
    - skills
    - projects
    - summary
    - certifications
    """
    sections: Dict[str, str] = {
        "summary": "",
        "experience": "",
        "education": "",
        "skills": "",
        "projects": "",
        "certifications": "",
        "other": ""
    }

    # Build regex pattern for headers
    # A header is usually on its own line, often uppercase or title-cased
    all_header_variants = []
    header_to_key = {}
    for key, headers in SECTION_HEADERS.items():
        for h in headers:
            all_header_variants.append(re.escape(h))
            header_to_key[h] = key

    pattern = rf"(?mi)^[ \t]*(?:[0-9]+\.?[ \t]*)?({'|'.join(all_header_variants)})[ \t]*(?::|--|—|\n|$)"
    matches = list(re.finditer(pattern, text))

    if not matches:
        # Fallback if line starts aren't detected: check case-insensitive match with colon or newline
        sections["other"] = text
        return sections

    # First segment before first recognized section header is typically Summary/Contact
    first_start = matches[0].start()
    if first_start > 0:
        sections["summary"] = text[:first_start].strip()

    for i, match in enumerate(matches):
        raw_header = match.group(1).lower().strip()
        sec_key = header_to_key.get(raw_header, "other")

        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start_pos:end_pos].strip()

        if sections[sec_key]:
            sections[sec_key] += "\n\n" + content
        else:
            sections[sec_key] = content

    return sections


def extract_degrees(text: str) -> List[str]:
    """Finds degrees mentioned in education section or full text."""
    found = []
    for pattern in DEGREE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            norm = m.strip().upper()
            if norm not in found:
                found.append(norm)
    return found


def extract_skills(text: str, custom_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extracts skills from text using phrase matching and the domain taxonomy.
    Returns:
    {
        "all_skills": List[str],          # unique list of matched skills (display case)
        "by_category": Dict[str, List[str]], # skills grouped by domain
        "count": int
    }
    """
    skill_list = custom_skills if custom_skills else ALL_SKILLS
    text_lower = f" {text.lower()} "

    # Normalize special characters for exact matching
    # e.g. handle punctuation around words
    clean_text = re.sub(r"[,;:\(\)\[\]\{\}\/]", " ", text_lower)
    clean_text = f" {re.sub(r'\s+', ' ', clean_text)} "

    matched_skills: Set[str] = set()

    for skill in skill_list:
        skill_clean = skill.lower().strip()
        # Word boundary match to prevent partial substring false positives (e.g. 'c' in 'car' or 'go' in 'good')
        if skill_clean in ["c", "r", "go"]:
            pattern = rf"(?:\b|\s){re.escape(skill_clean)}(?:\b|\s)"
        elif "/" in skill_clean or "." in skill_clean or "+" in skill_clean or "#" in skill_clean:
            pattern = rf"(?:\b|\s){re.escape(skill_clean)}(?:\b|\s)"
        else:
            pattern = rf"\b{re.escape(skill_clean)}\b"

        if re.search(pattern, clean_text):
            matched_skills.add(skill_clean)

    # Categorize matched skills
    by_category: Dict[str, List[str]] = {}
    for category, cat_skills in DOMAIN_SKILLS.items():
        cat_skills_lower = {s.lower() for s in cat_skills}
        cat_matches = [s for s in matched_skills if s in cat_skills_lower]
        if cat_matches:
            by_category[category] = sorted([s.title() if len(s) > 3 else s.upper() for s in cat_matches])

    # Format all_skills nicely
    formatted_skills = sorted([s.title() if len(s) > 3 and not s.isupper() else s.upper() for s in matched_skills])

    return {
        "all_skills": formatted_skills,
        "by_category": by_category,
        "count": len(formatted_skills)
    }


def parse_and_extract_all(raw_text: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts complete structured profile from resume text:
    name, contact info, sections, degrees, and skills.
    """
    contact = extract_contact_info(raw_text)
    name = extract_candidate_name(raw_text, filename)
    sections = extract_sections(raw_text)
    degrees = extract_degrees(sections.get("education", "") or raw_text)
    skills_data = extract_skills(raw_text)

    # Calculate section completeness
    has_skills_section = bool(sections.get("skills", "").strip())
    has_exp_section = bool(sections.get("experience", "").strip())
    has_edu_section = bool(sections.get("education", "").strip())

    completeness_score = (
        (0.35 if has_skills_section else 0.0) +
        (0.35 if has_exp_section else 0.0) +
        (0.30 if has_edu_section else 0.0)
    )

    return {
        "name": name,
        "filename": filename or "Unknown",
        "contact": contact,
        "sections": sections,
        "degrees": degrees,
        "skills": skills_data["all_skills"],
        "skills_by_category": skills_data["by_category"],
        "skill_count": skills_data["count"],
        "completeness_score": round(completeness_score, 2),
        "raw_text": raw_text
    }
