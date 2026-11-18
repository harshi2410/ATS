"""
Streamlit Web Application for AI-Powered ATS Resume Screener & Candidate Ranker.
Optimized for Desktop, Tablet, and Mobile devices with responsive CSS and touch-friendly controls.
Run with: streamlit run app.py
"""

import os
import glob
import io
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from src.parser import parse_resume
from src.extractor import parse_and_extract_all, extract_skills
from src.ranker import ATSRanker
from src.reporting import generate_ranking_dataframe, export_ranking_to_csv, generate_text_summary

# Page configuration
st.set_page_config(
    page_title="TalentMatch AI | Modern ATS Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto"
)

# Comprehensive Responsive & Mobile-First Styling
st.markdown("""
<style>
    /* Responsive Root & Fluid Typography */
    :root {
        --primary-blue: #2563EB;
        --primary-dark: #1E3A8A;
        --accent-teal: #0D9488;
        --bg-card: rgba(255, 255, 255, 0.95);
        --border-color: rgba(226, 232, 240, 0.8);
    }

    /* Container Spacing for Mobile & Desktop */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: clamp(1rem, 3vw, 3rem) !important;
        padding-right: clamp(1rem, 3vw, 3rem) !important;
        max-width: 1400px;
    }

    /* Responsive Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #0D9488 100%);
        border-radius: 16px;
        padding: clamp(1.2rem, 3vw, 2.2rem);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.2);
    }
    .hero-title {
        font-size: clamp(1.5rem, 3.5vw, 2.4rem);
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: clamp(0.85rem, 2vw, 1.05rem);
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
        font-weight: 400;
        line-height: 1.4;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.9rem;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: clamp(0.72rem, 1.5vw, 0.82rem);
        font-weight: 500;
    }

    /* Responsive Metric Grid (auto-fits 4 col on desktop, 2 col on tablet/phone, 1 col on small mobile) */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: clamp(8px, 2vw, 16px);
        margin: 1.2rem 0;
    }
    .metric-tile {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: clamp(10px, 2vw, 16px);
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.15s ease;
    }
    .metric-tile:hover {
        transform: translateY(-2px);
    }
    .metric-num {
        font-size: clamp(1.3rem, 2.5vw, 1.9rem);
        font-weight: 700;
        color: #1D4ED8;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: clamp(0.7rem, 1.5vw, 0.8rem);
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Candidate Mobile & Web Cards */
    .cand-card {
        background: var(--bg-card);
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: clamp(12px, 2vw, 18px);
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .cand-card:hover {
        border-color: #93C5FD;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.08);
    }
    .cand-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
    }
    .cand-name {
        font-size: clamp(1.05rem, 2vw, 1.25rem);
        font-weight: 700;
        color: #0F172A;
    }
    .score-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: clamp(0.85rem, 1.8vw, 1rem);
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .score-high {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .score-med {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
    }
    .score-low {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }

    /* Rank Badges */
    .rank-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    .rank-1 { background-color: #FEF08A; color: #854D0E; }
    .rank-2 { background-color: #E2E8F0; color: #475569; }
    .rank-3 { background-color: #FED7AA; color: #9A3412; }
    .rank-other { background-color: #F1F5F9; color: #64748B; }

    /* Pill Badges with Auto-Wrap */
    .badge-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin: 6px 0;
    }
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: clamp(0.72rem, 1.5vw, 0.82rem);
        font-weight: 600;
        white-space: normal;
        word-break: break-word;
    }
    .badge-matched {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .badge-missing {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .badge-extra {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }

    /* Touch-friendly buttons */
    div.stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 600;
        font-size: clamp(0.9rem, 2vw, 1rem);
        transition: all 0.2s ease;
    }

    /* Mobile media query overrides */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .hero-banner {
            border-radius: 12px;
            padding: 1.2rem;
        }
        .cand-header {
            flex-direction: column;
            align-items: flex-start;
        }
    }
</style>
""", unsafe_allow_html=True)


def load_preset_jds():
    """Loads bundled sample job descriptions."""
    presets = {}
    jd_files = glob.glob(os.path.join("sample_job_descriptions", "*.txt"))
    for f in jd_files:
        name = os.path.splitext(os.path.basename(f))[0].replace("_", " ").title()
        with open(f, "r", encoding="utf-8") as file:
            presets[name] = file.read().strip()
    return presets


def load_sample_resumes():
    """Loads bundled sample resume files."""
    files = glob.glob(os.path.join("sample_resumes", "*.pdf"))
    if not files:
        files = glob.glob(os.path.join("sample_resumes", "*.txt"))
    loaded = []
    for f in sorted(files):
        with open(f, "rb") as fl:
            loaded.append((os.path.basename(f), fl.read()))
    return loaded


# --- SIDEBAR CONFIGURATION (Collapsible & Touch-Optimized) ---
with st.sidebar:
    st.markdown("### ⚙️ ATS Engine Settings")
    st.caption("Customize match sensitivity and scoring weights.")

    with st.expander("⚖️ Scoring Algorithm Weights", expanded=True):
        tfidf_w = st.slider(
            "TF-IDF Context Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help="Measures overall text and domain vocabulary similarity."
        )
        skill_w = st.slider(
            "Skill Match Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
            help="Measures percentage of required technical skills present."
        )
        sec_w = st.slider(
            "Section Completeness Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
            help="Rewards resumes having Skills, Experience, and Education sections."
        )

    with st.expander("🔍 Filter & Thresholds", expanded=False):
        min_score_filter = st.slider("Minimum Match Threshold (%)", 0, 100, 0, 5)

    st.divider()
    st.caption("TalentMatch AI v2.0 • Offline & Secure")


# --- HERO BANNER (Responsive across phone, tablet, and desktop) ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎯 TalentMatch AI Screener</div>
    <div class="hero-subtitle">Automated resume parsing, NLP extraction, TF-IDF cosine similarity, and candidate ranking.</div>
    <div class="hero-badges">
        <span class="hero-pill">⚡ PyMuPDF & PyPDF</span>
        <span class="hero-pill">🧠 TF-IDF & Cosine Similarity</span>
        <span class="hero-pill">🏷️ N-gram Skill Matcher</span>
        <span class="hero-pill">📱 Mobile & Tablet Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --- MAIN TABS NAVIGATION ---
tab_screen, tab_results, tab_deepdive, tab_analytics, tab_guide = st.tabs([
    "📋 1. Setup & Screen",
    "🏆 2. Ranked Leaderboard",
    "🔍 3. Candidate Deep-Dive",
    "📊 4. Skill Analytics",
    "📖 5. How It Works"
])

preset_jds = load_preset_jds()
preset_names = list(preset_jds.keys()) + ["Custom Job Description"]


# ==============================================================================
# TAB 1: SETUP & SCREEN
# ==============================================================================
with tab_screen:
    col_jd, col_resumes = st.columns([1, 1], gap="medium")

    with col_jd:
        st.markdown("### 📋 Target Job Description")
        selected_preset = st.selectbox(
            "Choose a template or write your own:",
            options=preset_names,
            index=0 if preset_names else None
        )

        if selected_preset in preset_jds:
            default_jd_text = preset_jds[selected_preset]
        else:
            default_jd_text = "Paste your job description requirements and desired skills here..."

        jd_input = st.text_area(
            "Job Description Content:",
            value=default_jd_text,
            height=260,
            help="The screener extracts required domain skills and calculates cosine similarity against this text."
        )

        if jd_input.strip():
            extracted_jd_skills = extract_skills(jd_input)["all_skills"]
            with st.expander(f"✨ Detected Skills in Job Description ({len(extracted_jd_skills)})", expanded=False):
                if extracted_jd_skills:
                    badges_html = '<div class="badge-wrap">' + "".join([f'<span class="badge badge-extra">{s}</span>' for s in extracted_jd_skills]) + '</div>'
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.caption("No domain technical keywords detected in this text.")

    with col_resumes:
        st.markdown("### 📄 Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload Resumes (PDF or TXT):",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF or TXT files. Multiple files can be selected at once."
        )

        use_samples = st.toggle("⚡ Use 5 pre-loaded sample resumes (Alice, Bob, Carol, David, Eva)", value=not bool(uploaded_files))

        if use_samples:
            st.info("💡 Bundled sample resumes active: Alice (Senior ML), Bob (Full Stack), Carol (Data Analyst), David (DevOps), Eva (Junior Intern).")

    # Action Button
    st.write("")
    btn_screen = st.button("🚀 Analyze & Rank Resumes", type="primary", use_container_width=True)

    if btn_screen:
        # Collect resumes
        candidates_raw_data = []

        if uploaded_files:
            for up_file in uploaded_files:
                bytes_data = up_file.read()
                raw_text, err = parse_resume(bytes_data, filename=up_file.name)
                if raw_text:
                    profile = parse_and_extract_all(raw_text, filename=up_file.name)
                    candidates_raw_data.append(profile)
                else:
                    st.warning(f"Could not parse {up_file.name}: {err}")
        elif use_samples:
            samples = load_sample_resumes()
            for fname, content in samples:
                raw_text, _ = parse_resume(content, filename=fname)
                if raw_text:
                    profile = parse_and_extract_all(raw_text, filename=fname)
                    candidates_raw_data.append(profile)

        if not jd_input.strip():
            st.error("Please enter a Job Description.")
        elif not candidates_raw_data:
            st.error("Please upload resumes or enable pre-loaded samples.")
        else:
            with st.spinner("Parsing documents, vectorizing with TF-IDF, and ranking..."):
                ranker = ATSRanker(
                    tfidf_weight=tfidf_w,
                    skill_match_weight=skill_w,
                    section_weight=sec_w
                )
                ranked_results = ranker.rank_candidates(jd_input, candidates_raw_data)
                st.session_state["ranked_results"] = ranked_results
                st.session_state["jd_input"] = jd_input
                st.success(f"Successfully processed and ranked {len(ranked_results)} candidates! Switch to the '🏆 2. Ranked Leaderboard' tab to view results.")


# ==============================================================================
# TAB 2: RANKED LEADERBOARD
# ==============================================================================
with tab_results:
    if "ranked_results" in st.session_state and st.session_state["ranked_results"]:
        ranked = st.session_state["ranked_results"]
        filtered_ranked = [c for c in ranked if c["total_score"] >= min_score_filter]

        if not filtered_ranked:
            st.warning(f"No candidates met the minimum score threshold of {min_score_filter}%. Try lowering the threshold in the sidebar.")
        else:
            top_cand = filtered_ranked[0]
            avg_score = round(sum(c["total_score"] for c in filtered_ranked) / len(filtered_ranked), 1)

            # Responsive Metric Tiles
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="metric-num">{top_cand['name']}</div>
                    <div class="metric-lbl">🥇 Top Candidate</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-num">{top_cand['total_score']}%</div>
                    <div class="metric-lbl">🎯 Highest Score</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-num">{avg_score}%</div>
                    <div class="metric-lbl">📊 Average Match</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-num">{len(filtered_ranked)}</div>
                    <div class="metric-lbl">👥 Candidates</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick Export Toolbar
            col_exp1, col_exp2 = st.columns([1, 1], gap="small")
            df = generate_ranking_dataframe(filtered_ranked)
            with col_exp1:
                st.download_button(
                    label="📥 Download CSV Report",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="ats_candidate_ranking.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_exp2:
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json.dumps(filtered_ranked, indent=2).encode("utf-8"),
                    file_name="ats_candidate_ranking.json",
                    mime="application/json",
                    use_container_width=True
                )

            st.write("")

            # View switcher for Mobile vs Desktop
            view_mode = st.radio("Display View:", ["📱 Mobile-Friendly Card Feed", "💻 Standard Table View"], horizontal=True)

            if view_mode == "📱 Mobile-Friendly Card Feed":
                for c in filtered_ranked:
                    rank_class = "rank-1" if c['rank'] == 1 else ("rank-2" if c['rank'] == 2 else ("rank-3" if c['rank'] == 3 else "rank-other"))
                    score_class = "score-high" if c['total_score'] >= 50 else ("score-med" if c['total_score'] >= 25 else "score-low")

                    matched_html = "".join([f'<span class="badge badge-matched">{s}</span>' for s in c["matched_skills"][:6]])
                    if len(c["matched_skills"]) > 6:
                        matched_html += f'<span class="badge badge-matched">+{len(c["matched_skills"])-6} more</span>'

                    missing_html = "".join([f'<span class="badge badge-missing">{s}</span>' for s in c["missing_skills"][:4]])
                    if len(c["missing_skills"]) > 4:
                        missing_html += f'<span class="badge badge-missing">+{len(c["missing_skills"])-4} more</span>'

                    st.markdown(f"""
                    <div class="cand-card">
                        <div class="cand-header">
                            <div>
                                <span class="rank-tag {rank_class}">#{c['rank']}</span>
                                <span class="cand-name">{c['name']}</span>
                                <span style="font-size: 0.8rem; color: #64748B; margin-left: 6px;">({c['filename']})</span>
                            </div>
                            <span class="score-badge {score_class}">{c['total_score']}% Match</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #475569; margin: 4px 0 8px 0;">
                            <b>Skill Match:</b> {c['skill_match_score']}% &nbsp;|&nbsp;
                            <b>TF-IDF Sim:</b> {c['tfidf_score']}% &nbsp;|&nbsp;
                            <b>Degrees:</b> {', '.join(c['degrees']) if c['degrees'] else 'N/A'}
                        </div>
                        <div style="font-size: 0.82rem; font-weight: 600; color: #166534; margin-top: 6px;">✅ Matched Skills:</div>
                        <div class="badge-wrap">{matched_html if matched_html else '<span style="font-size:0.8rem; color:#94A3B8;">None</span>'}</div>
                        <div style="font-size: 0.82rem; font-weight: 600; color: #991B1B; margin-top: 6px;">❌ Missing Skills:</div>
                        <div class="badge-wrap">{missing_html if missing_html else '<span style="font-size:0.8rem; color:#94A3B8;">None</span>'}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.dataframe(
                    df[[
                        "Rank", "Candidate Name", "Total Match (%)",
                        "Skill Match (%)", "TF-IDF Similarity (%)",
                        "Matched Skills", "Missing Skills", "Degrees", "Email", "Filename"
                    ]],
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info("💡 No screening results yet. Configure a Job Description and click 'Analyze & Rank Resumes' in Tab 1!")


# ==============================================================================
# TAB 3: CANDIDATE DEEP-DIVE
# ==============================================================================
with tab_deepdive:
    if "ranked_results" in st.session_state and st.session_state["ranked_results"]:
        ranked = st.session_state["ranked_results"]
        candidate_options = [f"#{c['rank']} - {c['name']} ({c['total_score']}%)" for c in ranked]
        sel_idx = st.selectbox("Select Candidate to Inspect:", range(len(candidate_options)), format_func=lambda i: candidate_options[i])

        if sel_idx is not None and sel_idx < len(ranked):
            cand = ranked[sel_idx]

            col_info, col_skills = st.columns([1, 1], gap="medium")

            with col_info:
                st.markdown(f"### {cand['name']}")
                st.caption(f"📁 Source: `{cand['filename']}`")

                st.write(f"📧 **Email:** {cand['contact'].get('email') or 'Not listed'}")
                st.write(f"📞 **Phone:** {cand['contact'].get('phone') or 'Not listed'}")
                if cand['contact'].get('linkedin'):
                    st.write(f"🔗 **LinkedIn:** {cand['contact']['linkedin']}")
                if cand['contact'].get('github'):
                    st.write(f"🐙 **GitHub:** {cand['contact']['github']}")
                st.write(f"🎓 **Degrees Identified:** {', '.join(cand['degrees']) if cand['degrees'] else 'None detected'}")

                st.markdown("#### 🎯 Score Breakdown")
                st.progress(cand['total_score'] / 100.0, text=f"Total ATS Score: {cand['total_score']}%")
                st.progress(cand['skill_match_score'] / 100.0, text=f"Direct Skill Match: {cand['skill_match_score']}%")
                st.progress(cand['tfidf_score'] / 100.0, text=f"TF-IDF Context Similarity: {cand['tfidf_score']}%")
                st.progress(cand['completeness_score'] / 100.0, text=f"Section Completeness: {cand['completeness_score']}%")

                st.markdown("#### 🔑 Shared Key Vocabulary (TF-IDF)")
                if cand['top_matching_keywords']:
                    st.write(", ".join([f"`{k}`" for k in cand['top_matching_keywords']]))
                else:
                    st.caption("No significant overlapping keywords.")

            with col_skills:
                st.markdown("#### ✅ Matched Required Skills")
                if cand["matched_skills"]:
                    badges_html = '<div class="badge-wrap">' + "".join([f'<span class="badge badge-matched">{s}</span>' for s in cand["matched_skills"]]) + '</div>'
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.caption("No direct skills matched.")

                st.markdown("#### ❌ Missing Required Skills")
                if cand["missing_skills"]:
                    badges_html = '<div class="badge-wrap">' + "".join([f'<span class="badge badge-missing">{s}</span>' for s in cand["missing_skills"]]) + '</div>'
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.caption("All required skills found!")

                st.markdown("#### ➕ Additional Skills Found on Resume")
                if cand["additional_skills"]:
                    badges_html = '<div class="badge-wrap">' + "".join([f'<span class="badge badge-extra">{s}</span>' for s in cand["additional_skills"][:15]]) + '</div>'
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.caption("No additional domain skills.")

            # Section tabs
            st.divider()
            with st.expander("📄 Parsed Resume Content & Section Extraction", expanded=False):
                sec_tabs = st.tabs(["Experience", "Education", "Skills", "Summary", "Full Raw Text"])
                with sec_tabs[0]:
                    st.text(cand["sections"].get("experience") or "No Experience section extracted.")
                with sec_tabs[1]:
                    st.text(cand["sections"].get("education") or "No Education section extracted.")
                with sec_tabs[2]:
                    st.text(cand["sections"].get("skills") or "No Skills section extracted.")
                with sec_tabs[3]:
                    st.text(cand["sections"].get("summary") or "No Summary section extracted.")
                with sec_tabs[4]:
                    st.text(cand.get("raw_text", ""))
    else:
        st.info("💡 Run screening in Tab 1 to unlock candidate deep-dives.")


# ==============================================================================
# TAB 4: SKILL ANALYTICS
# ==============================================================================
with tab_analytics:
    if "ranked_results" in st.session_state and st.session_state["ranked_results"] and "jd_input" in st.session_state:
        ranked = st.session_state["ranked_results"]
        jd_skills = extract_skills(st.session_state["jd_input"])["all_skills"]

        st.markdown("### 📊 Skill Coverage Across Candidate Pool")
        st.caption("Shows which required skills are well-represented vs hard to find in the candidate pool.")

        # Compute skill coverage counts
        skill_counts = {}
        for s in jd_skills:
            skill_counts[s] = sum(1 for c in ranked if s in c["matched_skills"])

        if skill_counts:
            chart_df = pd.DataFrame([
                {"Skill": k, "Candidates With Skill": v, "Percentage": round((v / len(ranked)) * 100, 1)}
                for k, v in skill_counts.items()
            ]).sort_values(by="Candidates With Skill", ascending=True)

            fig = px.bar(
                chart_df,
                x="Candidates With Skill",
                y="Skill",
                orientation="h",
                text="Percentage",
                title=f"Coverage of Required Skills across {len(ranked)} Resumes",
                color="Candidates With Skill",
                color_continuous_scale="Blues",
                labels={"Candidates With Skill": "Number of Candidates", "Skill": "Required Skill"}
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(
                height=max(400, len(skill_counts) * 28),
                margin=dict(l=10, r=20, t=40, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Score Distribution Chart
            st.markdown("### 📈 Match Score Comparison")
            score_df = pd.DataFrame([
                {
                    "Candidate": f"#{c['rank']} {c['name']}",
                    "Total ATS Score": c['total_score'],
                    "Skill Match": c['skill_match_score'],
                    "TF-IDF Similarity": c['tfidf_score']
                }
                for c in ranked
            ])
            fig_score = px.bar(
                score_df,
                x="Candidate",
                y=["Total ATS Score", "Skill Match", "TF-IDF Similarity"],
                barmode="group",
                title="Score Breakdown Comparison by Candidate",
                labels={"value": "Score (%)", "variable": "Metric"}
            )
            fig_score.update_layout(margin=dict(l=10, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.info("💡 Run screening in Tab 1 to view interactive charts and pool analytics.")


# ==============================================================================
# TAB 5: HOW IT WORKS
# ==============================================================================
with tab_guide:
    st.markdown("### 🧠 How the AI ATS Screening Engine Works")

    st.markdown("""
    This ATS screener combines **Natural Language Processing (NLP)**, **TF-IDF Vector Space Modeling**, and **Domain Taxonomy Matching** to compute an objective match score:

    #### 1. Hybrid Scoring Formula
    The overall candidate score is a weighted combination of three distinct signals:
    $$\\text{ATS Score} = w_{\\text{tfidf}} \\times S_{\\text{tfidf}} + w_{\\text{skill}} \\times S_{\\text{skill}} + w_{\\text{sec}} \\times S_{\\text{sec}}$$

    - **TF-IDF Cosine Similarity ($S_{\\text{tfidf}}$)**:
      Measures semantic vocabulary overlap between the full job description and the resume using unigrams and bigrams $(1, 2)$ with sublinear term-frequency scaling to penalize keyword repetition.
    - **Skill Recall Score ($S_{\\text{skill}}$)**:
      Measures the proportion of required domain skills found on the resume:
      $$\\text{Recall} = \\frac{|\\text{Required Skills} \\cap \\text{Resume Skills}|}{|\\text{Required Skills}|}$$
    - **Section Completeness ($S_{\\text{sec}}$)**:
      Validates whether the resume contains properly formatted sections for Skills, Experience, and Education.

    #### 2. Technical Token Protection
    Standard NLP tokenizers often strip punctuation, turning `C++` into `C`, `Node.js` into `Node js`, or `CI/CD` into `CI CD`.
    Our preprocessor protects technical terms with dedicated placeholders during tokenization to preserve semantic identity.

    #### 3. Section Segmentation & Extraction
    The extractor parses text into logical blocks:
    - **Experience** (work history, achievements)
    - **Education** (academic credentials, degrees)
    - **Skills** (technical competencies, frameworks)
    - **Contact Info** (email, phone, LinkedIn, GitHub)
    """)
