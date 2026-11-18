"""
Configuration and Domain Knowledge Base for the ATS Resume Screening System.
Includes extensive skill taxonomies, section headers, and regex patterns.
"""

# Categorized Skill Database
DOMAIN_SKILLS = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "c", "javascript", "typescript", "golang", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "bash", "shell", "powershell",
        "sql", "html", "css", "html5", "css3", "sass", "matlab", "dart"
    ],
    "AI, Machine Learning & NLP": [
        "machine learning", "deep learning", "artificial intelligence", "nlp",
        "natural language processing", "computer vision", "generative ai", "llm",
        "large language models", "transformers", "pytorch", "tensorflow", "keras",
        "scikit-learn", "sklearn", "pandas", "numpy", "scipy", "hugging face",
        "huggingface", "opencv", "spacy", "nltk", "xgboost", "lightgbm", "langchain",
        "llamaindex", "rag", "retrieval augmented generation", "fine-tuning",
        "reinforcement learning", "bert", "gpt", "faiss", "vector database",
        "prompt engineering", "sentiment analysis", "named entity recognition", "ner"
    ],
    "Data Engineering & Analytics": [
        "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "cassandra",
        "sqlite", "snowflake", "bigquery", "redshift", "apache spark", "spark",
        "pyspark", "kafka", "apache kafka", "hadoop", "airflow", "apache airflow",
        "etl", "data pipelines", "tableau", "power bi", "powerbi", "excel",
        "data warehousing", "data modeling", "feature engineering", "dbt", "dask"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "microsoft azure", "gcp",
        "google cloud platform", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "ci/cd", "ci cd", "github actions", "gitlab ci",
        "linux", "unix", "helm", "prometheus", "grafana", "nginx", "apache",
        "serverless", "lambda", "ecs", "eks", "cloudformation"
    ],
    "Web & Backend Frameworks": [
        "react", "react.js", "angular", "vue", "vue.js", "next.js", "node.js",
        "express", "express.js", "django", "flask", "fastapi", "spring",
        "spring boot", "rest api", "restful api", "rest", "graphql", "grpc",
        "microservices", "tailwind", "tailwind css", "bootstrap", "redux"
    ],
    "Software Practices & Methodologies": [
        "agile", "scrum", "git", "github", "gitlab", "bitbucket", "jira",
        "system design", "object oriented programming", "oop", "unit testing",
        "pytest", "tdd", "test driven development", "code review", "design patterns",
        "clean code", "microservices architecture"
    ],
    "Soft Skills & Leadership": [
        "problem solving", "critical thinking", "communication", "leadership",
        "team management", "collaboration", "mentorship", "cross-functional",
        "presentation", "analytical skills"
    ]
}

# Flattened set of all skills (lowercase) for fast lookup
ALL_SKILLS = sorted(list({skill.lower() for sublist in DOMAIN_SKILLS.values() for skill in sublist}))

# Common Resume Section Headers
SECTION_HEADERS = {
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history", "career history"
    ],
    "education": [
        "education", "academic background", "educational background",
        "qualifications", "academic qualifications", "degrees"
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "key skills",
        "technologies", "skill set", "tools & technologies", "competencies"
    ],
    "projects": [
        "projects", "academic projects", "key projects", "personal projects"
    ],
    "summary": [
        "summary", "professional summary", "profile", "executive profile",
        "objective", "career objective", "about me"
    ],
    "certifications": [
        "certifications", "certificates", "licenses", "courses"
    ]
}

# Regex patterns for contact information & education degrees
EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_PATTERN = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
LINKEDIN_PATTERN = r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/in\/[a-zA-Z0-9_-]+"
GITHUB_PATTERN = r"(?:https?:\/\/)?(?:www\.)?github\.com\/[a-zA-Z0-9_-]+"

DEGREE_PATTERNS = [
    r"\b(?:ph\.?d|doctor\s+of\s+philosophy)\b",
    r"\b(?:m\.?s\.?|m\.?tech|master\s+of\s+(?:science|technology|engineering|business|arts|computer\s+applications))\b",
    r"\b(?:b\.?s\.?|b\.?tech|b\.?e\.?|bachelor\s+of\s+(?:science|technology|engineering|arts|computer\s+applications))\b",
    r"\b(?:bachelor|master|doctorate|diploma|associate)\b"
]

# Default Ranking Weights
DEFAULT_WEIGHTS = {
    "tfidf_weight": 0.50,       # 50% Overall content & context similarity (TF-IDF Cosine)
    "skill_match_weight": 0.40, # 40% Direct domain skill match ratio
    "section_weight": 0.10      # 10% Section completeness (Skills, Exp, Edu)
}
