from __future__ import annotations

import json
import re


def normalize_text(value):
    if not value:
        return ""

    return re.sub(r"\s+", " ", value).strip()


def join_nonempty(values):
    return ", ".join(
        item.strip()
        for item in values
        if item and item.strip()
    )


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )


def safe_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def build_student_search_text(
    *,
    name="",
    resume_text="",
    skills=None,
    projects=None,
    tools=None,
    target_roles=None,
):
    skills = skills or []
    projects = projects or []
    tools = tools or []
    target_roles = target_roles or []

    parts = [
        normalize_text(name),
        normalize_text(resume_text),
        f"Skills: {join_nonempty(skills)}",
        f"Projects: {join_nonempty(projects)}",
        f"Tools: {join_nonempty(tools)}",
        f"Target Roles: {join_nonempty(target_roles)}",
    ]

    return normalize_text(
        " | ".join(
            p for p in parts if p
        )
    )


def build_job_search_text(
    *,
    title="",
    company="",
    description="",
    skills=None,
    tools=None,
    location="",
    job_type="",
):
    skills = skills or []
    tools = tools or []

    parts = [
        f"Role: {title}",
        f"Company: {company}",
        normalize_text(description),
        f"Skills: {join_nonempty(skills)}",
        f"Tools: {join_nonempty(tools)}",
        f"Location: {location}",
        f"Type: {job_type}",
    ]

    return normalize_text(
        " | ".join(
            p for p in parts if p
        )
    )