# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.rag_agent import MeddraRAGAgent, RAGResponse


@st.cache_resource
def get_agent(config_text: str) -> MeddraRAGAgent:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    return MeddraRAGAgent(config_path)


def render_hierarchy(response: RAGResponse, lang_dict: dict) -> None:
    if not response.hierarchy:
        return
    hierarchy_records = []
    for level in ["LLT", "PT", "HLT", "HLGT", "SOC"]:
        data = response.hierarchy.get(level)
        if data:
            hierarchy_records.append(
                {
                    lang_dict["level"]: level,
                    lang_dict["code"]: data.get("code", ""),
                    lang_dict["term"]: data.get("term", ""),
                }
            )
    if hierarchy_records:
        st.subheader(lang_dict["hierarchy"])
        st.table(pd.DataFrame(hierarchy_records))
    else:
        st.info(lang_dict["no_hierarchy"])


def render_candidates(response: RAGResponse, lang_dict: dict) -> None:
    st.sidebar.header(lang_dict["vector_results"])
    if not response.candidates:
        st.sidebar.info(lang_dict["no_candidates"])
        return

    for idx, candidate in enumerate(response.candidates, start=1):
        lexical_pct = candidate.lexical_score * 100 if candidate.lexical_score else 0
        st.sidebar.markdown(
            f"**{idx}. {candidate.term}**\n\n"
            f"- {lang_dict['level']}: {candidate.level}\n"
            f"- {lang_dict['code']}: `{candidate.code}`\n"
            f"- {lang_dict['vector_score']}: {candidate.score:.3f}\n"
            f"- {lang_dict['lexical_match']}: {lexical_pct:.1f}%\n"
            f"- {lang_dict['combined_score']}: {candidate.combined_score:.3f}"
        )
        doc_text = candidate.document_text or candidate.metadata.get("document_text", "")
        if doc_text:
            st.sidebar.code(doc_text, language="text")

def main() -> None:
    st.set_page_config(page_title="MedDRA-Coding-AI", layout="wide")
    
    # 语言切换配置
    LANGUAGES = {
        "English": {
            "title": "MedDRA-Coding-AI",
            "caption": "Conversational coding assistant with Retrieval-Augmented Generation",
            "settings": "Settings",
            "language": "Language",
            "version": "MedDRA Version",
            "top_k": "Top-k Candidates",
            "input_label": "Enter a medical term",
            "submit_btn": "Encode Term",
            "info_msg": "Provide a medical term and click **Encode Term** to begin.",
            "spinner": "Running retrieval and reasoning...",
            "warning": "Low-confidence result. Consider refining the input term or reviewing the suggested alternatives.",
            "selected_code": "Selected MedDRA Code",
            "term": "Term",
            "code": "Code",
            "level": "Level",
            "hierarchy": "Hierarchy",
            "llm_reasoning": "LLM Reasoning",
            "raw_output": "Raw LLM Output",
            "no_index_error": "No MedDRA indexes were found. Run `python meddra_rag_assistant/build_index.py` to build vector indexes before using the assistant.",
            "vector_results": "Vector Search Results",
            "no_candidates": "No candidates retrieved.",
            "no_hierarchy": "No hierarchy information available for the selected code.",
            "vector_score": "Vector score",
            "lexical_match": "Lexical match",
            "combined_score": "Combined score"
        },
        "中文": {
            "title": "基于AI的MedDRA编码助手",
            "caption": "知足不辱，知止不殆，可以长久",
            "settings": "设置",
            "language": "语言",
            "version": "MedDRA 版本",
            "top_k": "检索结果数",
            "input_label": "输入医学术语",
            "submit_btn": "编码",
            "info_msg": "请提供医学术语并点击**编码**开始。",
            "spinner": "正在运行检索和推理...",
            "warning": "低置信度结果。建议优化输入术语或查看建议的替代方案。",
            "selected_code": "选定的 MedDRA 编码",
            "term": "术语",
            "code": "编码",
            "level": "级别",
            "hierarchy": "层级结构",
            "llm_reasoning": "LLM 推理过程",
            "raw_output": "原始 LLM 输出",
            "no_index_error": "未找到 MedDRA 索引。请运行 `python meddra_rag_assistant/build_index.py` 来构建向量索引后再使用助手。",
            "vector_results": "向量检索结果",
            "no_candidates": "未检索到候选项。",
            "no_hierarchy": "所选编码无可用的层级结构信息。",
            "vector_score": "向量得分",
            "lexical_match": "词汇匹配度",
            "combined_score": "综合得分"
        }
    }
    
    # 初始化语言选择
    if "language" not in st.session_state:
        st.session_state.language = "中文"
    
    config_path = Path(__file__).resolve().parent / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8")

    agent = get_agent(config_text)
    versions = agent.available_versions()
    
    # 获取当前语言文本
    lang = LANGUAGES[st.session_state.language]
    
    if not versions:
        st.error(lang["no_index_error"])
        return

    st.title(lang["title"])
    st.caption(lang["caption"])

    with st.sidebar:
        st.header(lang["settings"])
        
        # 语言选择器
        selected_lang = st.selectbox(
            lang["language"],
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.language),
            key="language_selector"
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
        
        selected_version = st.selectbox(lang["version"], versions, index=0, key="version_selector")
        default_top_k = int(agent.retrieval_config.get("top_k", 5))
        top_k = st.slider(lang["top_k"], min_value=3, max_value=15, value=default_top_k)

    with st.form(key="term_form"):
        term = st.text_input(lang["input_label"], "")
        submitted = st.form_submit_button(lang["submit_btn"])

    if not submitted or not term.strip():
        st.info(lang["info_msg"])
        return

    with st.spinner(lang["spinner"]):
        response = agent.run(term=term, version_key=selected_version, top_k=int(top_k))

    render_candidates(response, lang)

    if response.low_confidence:
        st.warning(lang["warning"])

    if response.best_match:
        st.subheader(lang["selected_code"])
        st.markdown(
            f"- **{lang['term']}:** {response.best_match.get('term', '')}\n"
            f"- **{lang['code']}:** `{response.best_match.get('code', '')}`\n"
            f"- **{lang['level']}:** {response.best_match.get('level', '')}"
        )
        doc_text = response.best_match.get("document_text")
        if doc_text:
            st.code(doc_text, language="text")

    if agent.include_hierarchy:
        render_hierarchy(response, lang)

    st.subheader(lang["llm_reasoning"])
    st.write(response.reasoning)

    with st.expander(lang["raw_output"]):
        st.code(response.llm_output or "", language="json")


if __name__ == "__main__":
    main()
