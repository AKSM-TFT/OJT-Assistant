"""
Tools:
  - scan_resume_tool        → Agent 1: extract data + flag errors
  - improve_resume_tool     → Agent 2: suggest improvements
  - interview_question_tool → Agent 3: generate behavioral interview question
  - find_internships_tool   → Agent 4: web-search for OJT near a location
  - cover_letter_tool       → Agent 5: write a tailored cover letter
  - rewrite_resume_tool     → Agent 6: produce a polished ATS-ready resume
"""
 
import json
from typing import Annotated
 
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
 
from llm import llm as my_llm

# Helper functions for LLM calls and JSON parsing
 
def _call_llm(system: str, user: str) -> str:
    """Single LLM call; returns raw text content."""
    from langchain_core.messages import SystemMessage, HumanMessage
    response = my_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content
 
 
def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
 
 
# Tool 1 — Resume Scanner 
 
SCANNER_SYSTEM = """You are a professional resume analyst with 15+ years of HR experience.
Scan the resume and return a JSON object with EXACTLY this structure:
{
  "personal_info": {"name":"","email":"","phone":"","location":"","linkedin":""},
  "summary": "",
  "education": [],
  "experience": [],
  "hard_skills": [],
  "soft_skills": [],
  "certifications": [],
  "languages": [],
  "errors": [
    {"type":"grammar|formatting|missing_info|inconsistency|spelling",
     "location":"section name","description":"","severity":"high|medium|low"}
  ],
  "overall_score": 0
}
Be thorough — check spelling, grammar, vague bullets, missing metrics, passive voice, date consistency.
Return ONLY valid JSON."""
 
 
@tool
def scan_resume_tool(
    resume_text: Annotated[str, "The full plain-text content of the resume to scan."]
) -> str:
    """
    Scan a resume for errors, extract all structured data (skills, experience,
    education), and score it from 0–100. Always call this first.
    """
    result = _call_llm(SCANNER_SYSTEM, f"Scan this resume:\n\n{resume_text}")
    return result  # return raw JSON string so the LLM can read it

# Tool 2 — Improvement Advisor
 
IMPROVER_SYSTEM = """You are an elite resume coach. Given a resume scan JSON, return improvement
recommendations as a JSON object with EXACTLY this structure:
{
  "priority_fixes": [
    {"issue":"","current":"","suggested":"","impact":"high|medium|low","why":""}
  ],
  "section_rewrites": {
    "summary": "improved version or null",
    "experience_bullets": [],
    "skills_additions": []
  },
  "ats_tips": [],
  "formatting_improvements": [],
  "overall_advice": "",
  "predicted_score_after_improvements": 0
}
Return ONLY valid JSON."""
 
 
@tool
def improve_resume_tool(
    scan_json: Annotated[str, "The JSON string returned by scan_resume_tool."]
) -> str:
    """
    Analyse the resume scan result and return concrete, prioritised improvements,
    ATS tips, section rewrites, and a predicted score after fixes.
    """
    return _call_llm(IMPROVER_SYSTEM, f"Here is the resume scan:\n\n{scan_json}")

# Tool 3 — HR Interviewer
 
INTERVIEWER_SYSTEM = """You are a sharp HR interviewer. Given resume data and optionally a
previous answer from the candidate, produce ONE behavioral interview question.
 
Rules:
- Tie the question to a SPECIFIC skill, experience, or achievement in the resume
- Use the STAR method (Situation, Task, Action, Result)
- If a previous answer is provided, drill deeper into it OR pivot to a new resume item
- Keep the question to 2–3 sentences maximum
- Mention the exact resume item by name
 
Return ONLY the question text, nothing else."""
 
 
@tool
def interview_question_tool(
    scan_json: Annotated[str, "The JSON string returned by scan_resume_tool."],
    previous_answer: Annotated[str, "The candidate's previous answer, or empty string for the first question."] = ""
) -> str:
    """
    Generate one behavioral interview question grounded in the candidate's resume.
    Pass the candidate's previous answer to get a follow-up, or leave empty for
    the opening question.
    """
    context = (
        f"Resume data:\n{scan_json}\n\n"
        f"Candidate's previous answer: {previous_answer if previous_answer else '(this is the first question)'}"
    )
    return _call_llm(INTERVIEWER_SYSTEM, context)

# Tool 4 — OJT / Internship Finder
 
_search = DuckDuckGoSearchRun()
 
 
@tool
def find_internships_tool(
    skills: Annotated[str, "Comma-separated list of the candidate's top skills."],
    location: Annotated[str, "The candidate's location, e.g. 'Angeles City, Pampanga'."],
) -> str:
    """
    Search the web for real OJT and internship opportunities that match the
    candidate's skills and location. Runs multiple searches and returns results.
    """
    skill_list = [s.strip() for s in skills.split(",")]
    queries = [
        f"internship {skill_list[0]} {location}",
        f"OJT {skill_list[1] if len(skill_list) > 1 else skill_list[0]} Philippines 2025",
        f"entry level {skill_list[0]} jobs {location}",
    ]
 
    results = []
    for q in queries:
        try:
            results.append(f"Query: {q}\n{_search.run(q)}")
        except Exception as e:
            results.append(f"Query: {q}\nError: {e}")
 
    return "\n\n---\n\n".join(results)

# Tool 5 — Cover Letter Generator
 
COVER_LETTER_SYSTEM = """You are a master cover letter writer. Write a tailored, compelling
cover letter. Rules:
- NEVER open with "I am writing to apply for..."
- Hook with a real achievement from the resume in the first sentence
- Connect 2–3 specific accomplishments to the target role
- Close with confident next-step language
- 3–4 tight paragraphs, under 350 words
- Professional but warm — sounds like a real person
 
Return the full cover letter text only, ready to send."""
 
 
@tool
def cover_letter_tool(
    scan_json: Annotated[str, "The JSON string returned by scan_resume_tool."],
    job_target: Annotated[str, "The role the candidate is applying for."],
    company: Annotated[str, "The target company name."] = "the company",
) -> str:
    """
    Write a tailored cover letter for the candidate based on their resume data,
    the target role, and the target company.
    """
    return _call_llm(
        COVER_LETTER_SYSTEM,
        f"Resume data:\n{scan_json}\n\nTarget Role: {job_target}\nTarget Company: {company}\n\nWrite the cover letter."
    )