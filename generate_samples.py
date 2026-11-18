"""
Generates realistic sample resumes in both TXT and PDF formats.
Uses ReportLab for PDF generation.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

SAMPLE_RESUMES = {
    "alice_chen_ml_expert": {
        "name": "Alice Chen",
        "title": "Lead Machine Learning & AI Engineer",
        "contact": "alice.chen@email.com | (415) 555-0192 | San Francisco, CA | linkedin.com/in/alicechen-ai | github.com/alicechen",
        "summary": "Innovative Lead Machine Learning Engineer with 6+ years of experience designing and deploying scalable deep learning, NLP, and generative AI systems. Expert in fine-tuning large language models (LLM), building transformer pipelines, and architecting production ML pipelines on AWS and Kubernetes.",
        "skills": "Python, PyTorch, TensorFlow, Scikit-Learn, Deep Learning, Natural Language Processing, NLP, Transformers, Hugging Face, LLM, RAG, Pandas, NumPy, SQL, Docker, Kubernetes, AWS, CI/CD, Git, Pytest, System Design",
        "experience": """Senior Machine Learning Engineer | Nexus AI Labs (2022 - Present)
• Architected and deployed retrieval-augmented generation (RAG) and LLM fine-tuning pipelines using PyTorch, Hugging Face, and LangChain, serving over 2M monthly queries.
• Built deep learning models for NLP classification and sentiment analysis, achieving 94.6% F1 score using Transformers and Scikit-Learn.
• Containerized AI microservices using Docker and orchestrated multi-GPU inference clusters on AWS EKS (Kubernetes) with automated CI/CD.

Machine Learning Engineer | Apex Data Systems (2019 - 2022)
• Developed automated data pipelines and feature engineering pipelines processing 50M+ daily records using Python, Pandas, and SQL.
• Trained and optimized XGBoost and TensorFlow models for recommendation systems, driving 18% lift in user engagement.
• Mentored 4 junior engineers in machine learning engineering, unit testing, and agile best practices.""",
        "education": """Master of Science in Computer Science (Artificial Intelligence Focus) | Stanford University (2019)
Bachelor of Science in Computer Engineering | UC Berkeley (2017)""",
        "projects": """• OpenRAG: Open-source high-throughput LLM retrieval engine built with PyTorch and FAISS.
• BioBERT-NER: Biomedical named entity recognition system utilizing transformer embeddings."""
    },

    "bob_martinez_fullstack": {
        "name": "Bob Martinez",
        "title": "Senior Full Stack Software Engineer",
        "contact": "bob.martinez@techmail.com | (212) 555-0144 | New York, NY | linkedin.com/in/bobmartinez-dev | github.com/bobmartinez",
        "summary": "Versatile Senior Full Stack Developer with 5+ years of experience creating robust, high-traffic web applications. Proven track record in architecting modern frontend interfaces with React and Next.js, and high-performance backend microservices with Node.js and TypeScript.",
        "skills": "JavaScript, TypeScript, React, Next.js, Node.js, Express, Redux, HTML5, CSS3, Tailwind CSS, PostgreSQL, MongoDB, Redis, REST API, GraphQL, Docker, Git, CI/CD, GitHub Actions, Agile, Scrum",
        "experience": """Senior Full Stack Developer | Horizon Web Solutions (2021 - Present)
• Spearheaded frontend modernization to React 18, Next.js, and TypeScript, improving Core Web Vitals and initial page load speed by 42%.
• Built distributed RESTful APIs and GraphQL endpoints in Node.js and Express handling 15,000 requests per minute with Redis caching.
• Designed relational database schemas in PostgreSQL and optimized complex SQL queries for 3x performance gains.
• Automated deployment workflows utilizing Docker and GitHub Actions CI/CD.

Full Stack Web Developer | Creative Pulse Digital (2019 - 2021)
• Engineered responsive web applications with React, Redux, and Tailwind CSS.
• Built backend CRUD services using Node.js and MongoDB; integrated third-party payment and auth APIs.
• Participated in agile sprints, daily standups, code reviews, and sprint retrospectives.""",
        "education": """Bachelor of Science in Software Engineering | University of Michigan (2019)""",
        "projects": """• DevStream: Real-time collaborative workspace built with Next.js, WebSockets, and Redis.
• TaskMaster Pro: Multi-tenant kanban board application using React and PostgreSQL."""
    },

    "carol_danvers_data_analyst": {
        "name": "Carol Danvers",
        "title": "Senior Data Analyst & BI Consultant",
        "contact": "carol.danvers@analytix.com | (312) 555-0188 | Chicago, IL | linkedin.com/in/caroldanvers-data | github.com/caroldanvers",
        "summary": "Results-oriented Senior Data Analyst with 5 years of experience delivering actionable business intelligence, enterprise dashboards, and predictive analytics. Advanced proficiency in SQL data extraction, Tableau/Power BI visualization, and Python analytical modeling.",
        "skills": "SQL, Python, Pandas, NumPy, Tableau, Power BI, Excel, Snowflake, BigQuery, Data Warehousing, ETL, Data Modeling, Feature Engineering, Matplotlib, Scikit-Learn, Business Intelligence, Problem Solving, Communication",
        "experience": """Senior Data Analyst | Global Insights Group (2021 - Present)
• Built and maintained 25+ executive dashboards in Tableau and Power BI tracking revenue, customer retention, and marketing ROI for C-suite leaders.
• Developed automated ETL pipelines in Python (Pandas) and SQL, migrating analytics infrastructure to Snowflake and reducing reporting lag by 60%.
• Conducted cohort analyses and statistical hypothesis testing to guide product roadmap decisions.

Data Analyst | Midwest FinTech Solutions (2019 - 2021)
• Analyzed customer transaction datasets exceeding 100M rows using complex SQL queries and window functions.
• Built financial modeling spreadsheets in advanced Excel (VBA, Power Query, pivot tables).
• Collaborated with engineering teams to ensure clean data ingestion into Google BigQuery.""",
        "education": """Master of Science in Business Analytics | Northwestern University (2019)
Bachelor of Science in Statistics and Mathematics | University of Illinois (2017)""",
        "projects": """• Customer Churn Predictor: Predictive customer classification model using Python and Scikit-Learn.
• Sales Performance Tracker: Interactive Power BI dashboard integrating multiple ERP data streams."""
    },

    "david_kim_devops": {
        "name": "David Kim",
        "title": "Senior Cloud Infrastructure & DevOps Engineer",
        "contact": "david.kim@cloudinfra.net | (206) 555-0177 | Seattle, WA | linkedin.com/in/davidkim-cloud | github.com/davidkim-ops",
        "summary": "Dedicated DevOps and Cloud Engineer with 5+ years of experience automating infrastructure, architecting resilient CI/CD pipelines, and managing containerized microservices in multi-cloud environments.",
        "skills": "AWS, Azure, Docker, Kubernetes, Terraform, Ansible, Jenkins, CI/CD, GitHub Actions, Linux, Bash, Python, Helm, Prometheus, Grafana, Microservices, Git, System Design, Problem Solving",
        "experience": """Senior DevOps Engineer | CloudScale Systems (2021 - Present)
• Provisioned and managed high-availability cloud infrastructure on AWS using Terraform (IaC) and Kubernetes (EKS).
• Built automated CI/CD deployment pipelines using Jenkins and GitHub Actions, cutting release turnaround times from 3 days to 25 minutes.
• Implemented cluster observability with Prometheus and Grafana dashboards, decreasing MTTR by 35%.

Site Reliability Engineer | Beacon Tech (2019 - 2021)
• Maintained Linux servers and containerized microservices with Docker across AWS and on-premise nodes.
• Wrote automation scripts in Python and Bash for database backups, log rotation, and security auditing.
• Administered Git version control workflows and managed access permissions across 80+ engineering repositories.""",
        "education": """Bachelor of Science in Computer Science | University of Washington (2019)""",
        "projects": """• KubeDeploy: Declarative CLI tool for zero-downtime Kubernetes deployments written in Python.
• MultiCloud-Terraform-Kit: Modular Terraform repository for automated VPC, IAM, and cluster provisioning."""
    },

    "eva_patel_junior": {
        "name": "Eva Patel",
        "title": "Junior Software Engineer / Developer Intern",
        "contact": "eva.patel@studentmail.edu | (512) 555-0133 | Austin, TX | linkedin.com/in/evapatel-cs | github.com/evapatel",
        "summary": "Enthusiastic and motivated Computer Science graduate with hands-on internship experience in software development. Strong foundational knowledge in Python, JavaScript, Git, and web technologies with an eagerness to learn modern ML and web frameworks.",
        "skills": "Python, JavaScript, HTML, CSS, Git, GitHub, SQL, SQLite, Object Oriented Programming, Unit Testing, Problem Solving, Communication",
        "experience": """Software Engineering Intern | BrightPath Tech (Summer 2023)
• Assisted engineering team in debugging and building REST API endpoints using Python and SQLite.
• Developed responsive frontend UI components using HTML5, CSS3, and vanilla JavaScript.
• Wrote unit tests and participated in sprint planning and daily standups.

Undergraduate Teaching Assistant | University Computer Science Dept (2022 - 2023)
• Tutored 60+ undergraduate students in introductory Python programming, data structures, and algorithms.
• Graded weekly coding assignments and helped students troubleshoot code errors and Git merge conflicts.""",
        "education": """Bachelor of Science in Computer Science | University of Texas at Austin (Graduated May 2024)""",
        "projects": """• Campus Book Exchange: Web app built with Python Flask and SQLite for student book trading.
• Weather Tracker: JavaScript application fetching live weather forecasts from external REST APIs."""
    }
}


def build_text_resume(data: dict) -> str:
    """Formats candidate data into clean text resume."""
    return f"""{data['name'].upper()}
{data['title']}
{data['contact']}

PROFESSIONAL SUMMARY
{data['summary']}

TECHNICAL SKILLS
{data['skills']}

WORK EXPERIENCE
{data['experience']}

EDUCATION
{data['education']}

KEY PROJECTS
{data['projects']}
"""


def build_pdf_resume(data: dict, output_path: str):
    """Compiles candidate data into a well-formatted PDF resume using ReportLab."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    name_style = ParagraphStyle(
        "ResumeName",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=2
    )
    title_style = ParagraphStyle(
        "ResumeTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=4
    )
    contact_style = ParagraphStyle(
        "ResumeContact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=10
    )
    heading_style = ParagraphStyle(
        "ResumeHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=8,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        "ResumeBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )

    story = []

    # Header
    story.append(Paragraph(data["name"], name_style))
    story.append(Paragraph(data["title"], title_style))
    story.append(Paragraph(data["contact"], contact_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=8))

    # Summary
    story.append(Paragraph("PROFESSIONAL SUMMARY", heading_style))
    story.append(Paragraph(data["summary"], body_style))

    # Skills
    story.append(Paragraph("TECHNICAL SKILLS", heading_style))
    story.append(Paragraph(data["skills"], body_style))

    # Experience
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", heading_style))
    for para in data["experience"].split("\n\n"):
        formatted = para.replace("\n", "<br/>")
        story.append(Paragraph(formatted, body_style))

    # Education
    story.append(Paragraph("EDUCATION", heading_style))
    for para in data["education"].split("\n"):
        story.append(Paragraph(para, body_style))

    # Projects
    story.append(Paragraph("KEY PROJECTS", heading_style))
    for para in data["projects"].split("\n"):
        story.append(Paragraph(para, body_style))

    doc.build(story)


def generate_all_samples():
    out_dir = "sample_resumes"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating sample resumes in '{out_dir}'...")
    for key, data in SAMPLE_RESUMES.items():
        # 1. TXT version
        txt_path = os.path.join(out_dir, f"{key}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(build_text_resume(data))
        print(f"  [+] Created {txt_path}")

        # 2. PDF version
        pdf_path = os.path.join(out_dir, f"{key}.pdf")
        build_pdf_resume(data, pdf_path)
        print(f"  [+] Created {pdf_path}")

    print("Sample generation complete!")


if __name__ == "__main__":
    generate_all_samples()
