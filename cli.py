"""
Command Line Interface (CLI) for the AI-Powered ATS Resume Screener.
Runs resume parsing, skill extraction, TF-IDF cosine similarity, and candidate ranking.
"""

import os
import sys
import glob
import argparse
from typing import List, Dict, Any

from src.parser import parse_resume
from src.extractor import parse_and_extract_all
from src.ranker import ATSRanker
from src.reporting import generate_ranking_dataframe, export_ranking_to_csv, export_ranking_to_json, generate_text_summary


def load_job_description(jd_path: str) -> str:
    """Loads job description from file."""
    if not os.path.exists(jd_path):
        raise FileNotFoundError(f"Job description file not found: {jd_path}")
    with open(jd_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def load_and_parse_resumes(resume_paths: List[str]) -> List[Dict[str, Any]]:
    """Parses all resumes into structured candidate profiles."""
    parsed_candidates = []
    for path in resume_paths:
        filename = os.path.basename(path)
        raw_text, err = parse_resume(path, filename=filename)
        if err and not raw_text:
            print(f"  [!] Warning: Failed to parse '{filename}': {err}", file=sys.stderr)
            continue
        profile = parse_and_extract_all(raw_text, filename=filename)
        parsed_candidates.append(profile)
    return parsed_candidates


def run_ats_pipeline(
    jd_path: str,
    resume_sources: List[str],
    output_csv: str = None,
    output_json: str = None,
    tfidf_weight: float = 0.50,
    skill_weight: float = 0.40,
    section_weight: float = 0.10,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Executes the complete ATS screening pipeline."""
    print("=" * 75)
    print("🤖 AI-POWERED ATS RESUME SCREENER & RANKING ENGINE")
    print("=" * 75)

    # 1. Load Job Description
    print(f"[1/4] Loading Job Description from: {jd_path}")
    jd_text = load_job_description(jd_path)
    print(f"      Job description loaded ({len(jd_text.split())} words)")

    # 2. Gather Resume Files
    resume_files = []
    for src in resume_sources:
        if os.path.isdir(src):
            # Prefer PDF files, or TXT if no PDF duplicate
            pdf_files = glob.glob(os.path.join(src, "*.pdf"))
            txt_files = glob.glob(os.path.join(src, "*.txt"))
            # If both exist, deduplicate by stem
            seen_stems = set()
            for p in pdf_files:
                stem = os.path.splitext(os.path.basename(p))[0]
                seen_stems.add(stem)
                resume_files.append(p)
            for t in txt_files:
                stem = os.path.splitext(os.path.basename(t))[0]
                if stem not in seen_stems:
                    resume_files.append(t)
        elif os.path.isfile(src):
            resume_files.append(src)
        else:
            matches = glob.glob(src)
            resume_files.extend(matches)

    resume_files = sorted(list(set(resume_files)))
    if not resume_files:
        print("[!] No resume files found to screen.")
        return []

    print(f"[2/4] Parsing and extracting sections from {len(resume_files)} resumes...")
    candidates = load_and_parse_resumes(resume_files)
    print(f"      Successfully extracted structured data for {len(candidates)} candidates.")

    # 3. Rank Candidates
    print(f"[3/4] Vectorizing with TF-IDF, computing Cosine Similarity & Skill Match...")
    ranker = ATSRanker(
        tfidf_weight=tfidf_weight,
        skill_match_weight=skill_weight,
        section_weight=section_weight
    )
    ranked = ranker.rank_candidates(jd_text, candidates)

    # 4. Display Results
    print(f"[4/4] Ranking complete! Top Candidates:\n")

    # Format table header
    header_fmt = "{:<5} {:<18} {:<10} {:<10} {:<10} {:<24}"
    row_fmt = "{:<5} {:<18} {:<10.1f}% {:<10.1f}% {:<10.1f}% {:<24}"
    print("-" * 80)
    print(header_fmt.format("Rank", "Candidate", "Total ATS", "Skill Sim", "TF-IDF", "Matched Skills"))
    print("-" * 80)

    for c in ranked:
        matched_str = ", ".join(c["matched_skills"][:3])
        if len(c["matched_skills"]) > 3:
            matched_str += f" (+{len(c['matched_skills'])-3})"
        print(row_fmt.format(
            f"#{c['rank']}",
            c['name'][:17],
            c['total_score'],
            c['skill_match_score'],
            c['tfidf_score'],
            matched_str[:24]
        ))
    print("-" * 80)

    # Export if requested
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        export_ranking_to_csv(ranked, output_csv)
        print(f"\n[+] Candidate Ranking Report saved to CSV: {output_csv}")

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        export_ranking_to_json(ranked, output_json)
        print(f"[+] Full JSON Report saved to: {output_json}")

    # Verbose summaries
    if verbose:
        print("\n" + "#" * 80)
        print("DETAILED CANDIDATE AUDIT REPORTS")
        print("#" * 80 + "\n")
        for c in ranked:
            print(generate_text_summary(c))

    return ranked


def main():
    parser = argparse.ArgumentParser(description="AI Resume Screener & ATS Matcher CLI")
    parser.add_argument("--jd", required=True, help="Path to Job Description text file")
    parser.add_argument("--resumes", required=True, nargs="+", help="Directory or file paths to candidate resumes (PDF/TXT)")
    parser.add_argument("--output", default="output/candidate_ranking_report.csv", help="Path to export CSV report")
    parser.add_argument("--json", default=None, help="Path to export JSON report")
    parser.add_argument("--tfidf-weight", type=float, default=0.50, help="Weight for TF-IDF cosine similarity (0.0-1.0)")
    parser.add_argument("--skill-weight", type=float, default=0.40, help="Weight for direct skill match ratio (0.0-1.0)")
    parser.add_argument("--section-weight", type=float, default=0.10, help="Weight for section completeness (0.0-1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed report cards for each candidate")

    args = parser.parse_args()

    run_ats_pipeline(
        jd_path=args.jd,
        resume_sources=args.resumes,
        output_csv=args.output,
        output_json=args.json,
        tfidf_weight=args.tfidf_weight,
        skill_weight=args.skill_weight,
        section_weight=args.section_weight,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
