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

# ── shared helper ─────────────────────────────────────────────────────────────
 
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