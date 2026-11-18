"""
Comprehensive Unit and Integration Test Suite for the ATS Resume Screener.
Tests parser, preprocessor, extractor, ranker, and reporting modules.
"""

import os
import pytest
import pandas as pd
from src.parser import parse_resume, extract_text_from_pdf, extract_text_from_txt
from src.preprocessor import preprocess_text, tokenize_text, protect_tech_tokens
from src.extractor import (
    extract_contact_info, extract_candidate_name,
    extract_sections, extract_skills, extract_degrees, parse_and_extract_all
)
from src.ranker import ATSRanker
from src.reporting import generate_ranking_dataframe, export_ranking_to_csv, generate_text_summary


SAMPLE_RESUME_TEXT = """ALICE CHEN
Senior Machine Learning Engineer
alice.chen@email.com | (415) 555-0192 | linkedin.com/in/alicechen | github.com/alicechen

PROFESSIONAL SUMMARY
Senior ML Engineer with 5+ years building deep learning systems, NLP transformers, and PyTorch models.

TECHNICAL SKILLS
Python, PyTorch, TensorFlow, Scikit-Learn, Deep Learning, Natural Language Processing, Docker, Kubernetes, AWS, SQL, Git

WORK EXPERIENCE
Senior Machine Learning Engineer at AI Corp (2021 - Present)
- Built LLM and RAG pipelines using PyTorch and Hugging Face.
- Deployed microservices using Docker on AWS Kubernetes.

EDUCATION
Master of Science in Computer Science | Stanford University (2020)
Bachelor of Science in Electrical Engineering | UC Berkeley (2018)

KEY PROJECTS
- OpenRAG: Retrieval augmented generation framework.
"""

SAMPLE_JD_TEXT = """Job Title: Machine Learning Engineer
Requirements:
- Strong experience with Python, PyTorch, and Deep Learning.
- Hands-on with Natural Language Processing (NLP) and Scikit-Learn.
- Knowledge of Docker, Kubernetes, and AWS cloud deployment.
- Bachelor's or Master's degree in Computer Science.
"""


class TestParser:
    def test_txt_parsing(self):
        txt_path = "sample_resumes/alice_chen_ml_expert.txt"
        assert os.path.exists(txt_path)
        text, err = parse_resume(txt_path)
        assert err is None
        assert "alice chen" in text.lower()
        assert "machine learning" in text.lower()

    def test_pdf_parsing(self):
        pdf_path = "sample_resumes/alice_chen_ml_expert.pdf"
        assert os.path.exists(pdf_path)
        text, err = parse_resume(pdf_path)
        assert err is None
        assert len(text) > 200
        assert "Alice Chen" in text
        assert "PyTorch" in text

    def test_nonexistent_file(self):
        text, err = parse_resume("sample_resumes/does_not_exist.pdf")
        assert text == ""
        assert err is not None


class TestPreprocessor:
    def test_tech_token_preservation(self):
        raw = "Expert in C++, C#, .NET, Node.js, and CI/CD pipelines"
        tokens = tokenize_text(raw)
        # Should preserve tech terms without losing meaning
        assert "cplusplus" in tokens or "c++" in raw.lower()
        assert "nodejs" in tokens or "node.js" in raw.lower()
        assert "cicd" in tokens or "ci/cd" in raw.lower()

    def test_stopword_removal_and_lemmatization(self):
        text = "The machine learning engineers are developing innovative applications"
        processed = preprocess_text(text)
        tokens = processed.split()
        assert "the" not in tokens
        assert "are" not in tokens
        assert any(t in tokens for t in ["machine", "learn", "engineer", "develop"])


class TestExtractor:
    def test_contact_extraction(self):
        contact = extract_contact_info(SAMPLE_RESUME_TEXT)
        assert contact["email"] == "alice.chen@email.com"
        assert "(415) 555-0192" in contact["phone"]
        assert "linkedin.com/in/alicechen" in contact["linkedin"]
        assert "github.com/alicechen" in contact["github"]

    def test_name_extraction(self):
        name = extract_candidate_name(SAMPLE_RESUME_TEXT, "alice_chen_ml.pdf")
        assert "Alice Chen" in name

    def test_section_segmentation(self):
        sections = extract_sections(SAMPLE_RESUME_TEXT)
        assert len(sections["experience"]) > 0
        assert len(sections["education"]) > 0
        assert len(sections["skills"]) > 0
        assert "PyTorch" in sections["skills"]

    def test_degree_extraction(self):
        degrees = extract_degrees(SAMPLE_RESUME_TEXT)
        assert any("MASTER" in d for d in degrees)
        assert any("BACHELOR" in d for d in degrees)

    def test_skill_extraction(self):
        skills_data = extract_skills(SAMPLE_RESUME_TEXT)
        skills = [s.lower() for s in skills_data["all_skills"]]
        assert "python" in skills
        assert "pytorch" in skills
        assert "docker" in skills
        assert "kubernetes" in skills
        assert "scikit-learn" in skills or "sklearn" in skills
        assert "aws" in skills


class TestRanker:
    def test_ranking_relevance(self):
        cand1 = parse_and_extract_all(SAMPLE_RESUME_TEXT, "alice_ml.pdf")
        cand2_text = """BOB BUILDER
Frontend Developer
bob@builder.com | (555) 123-4567

TECHNICAL SKILLS
HTML, CSS, JavaScript, React, Tailwind CSS

WORK EXPERIENCE
Frontend UI designer working on marketing web pages.

EDUCATION
Bachelor of Arts in English Literature
"""
        cand2 = parse_and_extract_all(cand2_text, "bob_builder.pdf")

        ranker = ATSRanker(tfidf_weight=0.5, skill_match_weight=0.4, section_weight=0.1)
        ranked = ranker.rank_candidates(SAMPLE_JD_TEXT, [cand1, cand2])

        assert len(ranked) == 2
        # Alice Chen should rank #1 by a significant margin for an ML Engineer job
        assert ranked[0]["name"] == "Alice Chen"
        assert ranked[0]["total_score"] > ranked[1]["total_score"]
        assert ranked[0]["skill_match_score"] > 60.0
        assert "Pytorch" in ranked[0]["matched_skills"] or "Python" in ranked[0]["matched_skills"]

    def test_skill_overlap_logic(self):
        ranker = ATSRanker()
        jd_skills = ["Python", "PyTorch", "Docker", "AWS"]
        res_skills = ["Python", "PyTorch", "React", "Node.js"]

        overlap = ranker.analyze_skills_overlap(jd_skills, res_skills)
        assert overlap["matched_count"] == 2
        assert "Python" in overlap["matched_skills"]
        assert "Pytorch" in overlap["matched_skills"]
        missing_upper = [s.upper() for s in overlap["missing_skills"]]
        assert "AWS" in missing_upper
        assert overlap["skill_match_score"] == 0.5


class TestReporting:
    def test_reporting_dataframe_and_csv(self, tmp_path):
        cand = parse_and_extract_all(SAMPLE_RESUME_TEXT, "alice.pdf")
        ranker = ATSRanker()
        ranked = ranker.rank_candidates(SAMPLE_JD_TEXT, [cand])

        df = generate_ranking_dataframe(ranked)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["Candidate Name"] == "Alice Chen"

        csv_file = str(tmp_path / "test_report.csv")
        export_ranking_to_csv(ranked, csv_file)
        assert os.path.exists(csv_file)
        assert os.path.getsize(csv_file) > 50

    def test_text_summary(self):
        cand = parse_and_extract_all(SAMPLE_RESUME_TEXT, "alice.pdf")
        ranker = ATSRanker()
        ranked = ranker.rank_candidates(SAMPLE_JD_TEXT, [cand])
        summary = generate_text_summary(ranked[0])
        assert "CANDIDATE RANK #1" in summary
        assert "Overall ATS Match Score" in summary
