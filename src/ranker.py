"""
Candidate Ranking and Similarity Scoring Engine.
Computes:
1. TF-IDF Vectorization with unigrams and bigrams
2. Cosine Similarity between Job Description and Resume Profiles
3. Keyword / Skill Overlap and Recall Analysis (Matched vs Missing Skills)
4. Composite ATS Match Score with customizable weighting
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DEFAULT_WEIGHTS
from src.preprocessor import preprocess_text
from src.extractor import extract_skills, parse_and_extract_all


class ATSRanker:
    def __init__(
        self,
        tfidf_weight: float = DEFAULT_WEIGHTS["tfidf_weight"],
        skill_match_weight: float = DEFAULT_WEIGHTS["skill_match_weight"],
        section_weight: float = DEFAULT_WEIGHTS["section_weight"]
    ):
        # Normalize weights so they sum to 1.0
        total = tfidf_weight + skill_match_weight + section_weight
        if total <= 0:
            total = 1.0
        self.tfidf_weight = tfidf_weight / total
        self.skill_match_weight = skill_match_weight / total
        self.section_weight = section_weight / total

        # TF-IDF Vectorizer with unigrams & bigrams and sublinear TF scaling
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=1
        )

    def analyze_skills_overlap(
        self,
        jd_skills: List[str],
        resume_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates skill recall, overlap, matched skills, and missing skills.
        """
        jd_set = set(s.lower() for s in jd_skills)
        res_set = set(s.lower() for s in resume_skills)

        matched_set = jd_set.intersection(res_set)
        missing_set = jd_set - res_set
        additional_set = res_set - jd_set

        # Skill match score based on recall of required skills
        if len(jd_set) > 0:
            recall_score = len(matched_set) / len(jd_set)
        else:
            recall_score = 1.0 if len(res_set) > 0 else 0.5

        # Jaccard index
        union_len = len(jd_set.union(res_set))
        jaccard_score = (len(matched_set) / union_len) if union_len > 0 else 0.0

        # Helper to format nicely
        def format_skill(s: str) -> str:
            return s.title() if len(s) > 3 and not s.isupper() else s.upper()

        return {
            "matched_skills": sorted([format_skill(s) for s in matched_set]),
            "missing_skills": sorted([format_skill(s) for s in missing_set]),
            "additional_skills": sorted([format_skill(s) for s in additional_set]),
            "matched_count": len(matched_set),
            "required_count": len(jd_set),
            "skill_match_score": round(recall_score, 4),
            "jaccard_score": round(jaccard_score, 4)
        }

    def rank_candidates(
        self,
        job_description: str,
        resumes_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ranks a list of parsed resume profiles against a job description.

        resumes_data: list of dicts produced by parse_and_extract_all()
        Returns list of ranked candidate dictionaries sorted by total_score descending.
        """
        if not resumes_data:
            return []

        # 1. Preprocess Job Description
        jd_clean = preprocess_text(job_description)
        jd_skills_data = extract_skills(job_description)
        jd_skills = jd_skills_data["all_skills"]

        # 2. Preprocess all resumes
        corpus = [jd_clean]
        cleaned_resumes = []
        for res in resumes_data:
            c_text = preprocess_text(res.get("raw_text", ""))
            cleaned_resumes.append(c_text)
            corpus.append(c_text)

        # 3. Compute TF-IDF Vectors & Cosine Similarities
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        jd_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]

        cosine_sims = cosine_similarity(jd_vector, resume_vectors)[0]
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        # Top JD keywords for context
        jd_dense = jd_vector.toarray()[0]
        top_jd_indices = jd_dense.argsort()[-15:][::-1]
        top_jd_keywords = [
            feature_names[i] for i in top_jd_indices if jd_dense[i] > 0
        ]

        # 4. Score each candidate
        ranked_list = []
        for idx, res in enumerate(resumes_data):
            cos_sim = float(cosine_sims[idx])
            # Bound cosine similarity to [0.0, 1.0]
            cos_sim = max(0.0, min(1.0, cos_sim))

            # Skill overlap
            res_skills = res.get("skills", [])
            overlap_info = self.analyze_skills_overlap(jd_skills, res_skills)
            skill_score = overlap_info["skill_match_score"]

            # Section completeness
            completeness = res.get("completeness_score", 0.8)

            # Composite ATS Score
            composite_score = (
                (self.tfidf_weight * cos_sim) +
                (self.skill_match_weight * skill_score) +
                (self.section_weight * completeness)
            )

            # Percentage (0 - 100)
            final_percentage = round(composite_score * 100, 2)
            tfidf_percentage = round(cos_sim * 100, 2)
            skill_percentage = round(skill_score * 100, 2)

            # Find candidate's top matching keywords with JD
            res_dense = resume_vectors[idx].toarray()[0]
            # Multiply term weights to find shared significant terms
            shared_weights = jd_dense * res_dense
            top_shared_indices = shared_weights.argsort()[-8:][::-1]
            shared_keywords = [
                feature_names[i] for i in top_shared_indices if shared_weights[i] > 0
            ]

            candidate_summary = {
                "rank": 0,  # Assigned after sorting
                "name": res.get("name", "Unknown"),
                "filename": res.get("filename", "Unknown"),
                "total_score": final_percentage,
                "tfidf_score": tfidf_percentage,
                "skill_match_score": skill_percentage,
                "completeness_score": round(completeness * 100, 1),
                "matched_skills": overlap_info["matched_skills"],
                "missing_skills": overlap_info["missing_skills"],
                "additional_skills": overlap_info["additional_skills"],
                "all_extracted_skills": res_skills,
                "degrees": res.get("degrees", []),
                "contact": res.get("contact", {}),
                "sections": res.get("sections", {}),
                "top_matching_keywords": shared_keywords,
                "raw_text": res.get("raw_text", "")
            }
            ranked_list.append(candidate_summary)

        # Sort descending by total score
        ranked_list.sort(key=lambda x: x["total_score"], reverse=True)

        # Assign ranks
        for i, cand in enumerate(ranked_list):
            cand["rank"] = i + 1

        return ranked_list
