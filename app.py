from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import GROQ_API_KEY, SEMANTIC_SCHOLAR_API_KEY
from src.evaluation.manual import cards_to_manual_csv_bytes
from src.pipeline.run_pipeline import run_contradiction_pipeline
from src.schemas import Claim, ClaimPair, ConflictCard, ContradictionScore, Paper
from src.utils.io import read_bytes_if_exists

st.set_page_config(
    page_title="Contradiction Finder for Research Reading",
    page_icon=":material/manage_search:",
    layout="wide",
)


def as_records(items: list[Any]) -> list[dict[str, Any]]:
    records = []
    for item in items:
        if hasattr(item, "model_dump"):
            records.append(item.model_dump(mode="json"))
        elif hasattr(item, "dict"):
            records.append(item.dict())
        else:
            records.append(dict(item))
    return records


def paper_lookup(papers: list[Paper]) -> dict[str, Paper]:
    return {paper.paper_id: paper for paper in papers}


def render_metrics(result: dict[str, Any]) -> None:
    counts = result.get("stats", {}).get("counts", {})
    with st.container(horizontal=True):
        st.metric("Papers", counts.get("papers", 0), border=True)
        st.metric("Sentences", counts.get("sentences", 0), border=True)
        st.metric("Claims", counts.get("claims", 0), border=True)
        st.metric("Candidate pairs", counts.get("candidate_pairs", 0), border=True)
        st.metric(
            "Possible contradictions",
            counts.get("possible_contradictions", 0),
            border=True,
        )
        st.metric("Conflict cards", counts.get("conflict_cards", 0), border=True)


def render_papers(papers: list[Paper]) -> None:
    if not papers:
        st.caption("No retrieved papers to display.")
        return

    rows = []
    for paper in papers:
        rows.append(
            {
                "Title": paper.title,
                "Year": paper.year,
                "Venue": paper.venue,
                "Authors": ", ".join(paper.authors[:5]),
                "Citations": paper.citation_count,
                "URL": paper.url,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={"URL": st.column_config.LinkColumn("URL")},
    )

    if st.toggle("Show abstracts", value=False):
        for paper in papers:
            label = f"{paper.title} ({paper.year or 'year unknown'})"
            with st.expander(label, icon=":material/article:"):
                st.write(paper.abstract)


def render_claims(claims: list[Claim], papers: list[Paper]) -> None:
    if not claims:
        st.caption("No extracted claims to display.")
        return

    papers_by_id = paper_lookup(papers)
    rows = []
    for claim in claims:
        paper = papers_by_id.get(claim.paper_id)
        rows.append(
            {
                "Claim": claim.normalized_claim,
                "Paper": paper.title if paper else claim.paper_id,
                "Type": claim.claim_type,
                "Confidence": claim.confidence,
                "Method": claim.extraction_method,
                "Original sentence": claim.original_sentence,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        },
    )


def render_pairs(
    pairs: list[ClaimPair],
    claims: list[Claim],
    papers: list[Paper],
    scores: list[ContradictionScore],
) -> None:
    if not pairs:
        st.caption("No candidate claim pairs to display.")
        return

    papers_by_id = paper_lookup(papers)
    claims_by_id = {claim.claim_id: claim for claim in claims}
    scores_by_pair = {score.pair_id: score for score in scores}
    rows = []
    for pair in pairs:
        score = scores_by_pair.get(pair.pair_id)
        paper_a = papers_by_id.get(pair.paper_a_id)
        paper_b = papers_by_id.get(pair.paper_b_id)
        claim_a = claims_by_id.get(pair.claim_a_id)
        claim_b = claims_by_id.get(pair.claim_b_id)
        rows.append(
            {
                "Claim A": pair.claim_a,
                "Claim B": pair.claim_b,
                "Similarity": pair.similarity_score,
                "NLI contradiction": score.contradiction_score if score else None,
                "NLI label": score.predicted_label if score else None,
                "Paper A": paper_a.title if paper_a else pair.paper_a_id,
                "Paper B": paper_b.title if paper_b else pair.paper_b_id,
                "Evidence A": claim_a.original_sentence if claim_a else "",
                "Evidence B": claim_b.original_sentence if claim_b else "",
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Similarity": st.column_config.ProgressColumn(
                "Similarity",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "NLI contradiction": st.column_config.ProgressColumn(
                "NLI contradiction",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        },
    )


def render_conflict_cards(cards: list[ConflictCard]) -> None:
    if not cards:
        st.caption("No conflict cards met the selected contradiction threshold.")
        return

    for card in cards:
        with st.container(border=True):
            st.badge(
                "Possible contradiction",
                icon=":material/warning:",
                color="orange",
            )
            st.subheader(card.topic_title)
            st.caption(
                f"NLI contradiction score: {card.contradiction_score:.2f} | "
                f"Reason category: {card.possible_reason_category}"
            )
            left, right = st.columns(2)
            with left:
                st.markdown("**Claim A**")
                st.write(card.claim_a_text)
                st.markdown("**Evidence A**")
                st.write(card.evidence_a)
                st.caption(f"{card.paper_a_title} ({card.paper_a_year or 'year unknown'})")
                if card.paper_a_url:
                    st.link_button(
                        "Open paper A",
                        card.paper_a_url,
                        icon=":material/open_in_new:",
                    )
            with right:
                st.markdown("**Claim B**")
                st.write(card.claim_b_text)
                st.markdown("**Evidence B**")
                st.write(card.evidence_b)
                st.caption(f"{card.paper_b_title} ({card.paper_b_year or 'year unknown'})")
                if card.paper_b_url:
                    st.link_button(
                        "Open paper B",
                        card.paper_b_url,
                        icon=":material/open_in_new:",
                    )
            st.markdown("**Cautious interpretation**")
            st.write(card.explanation)


def render_exports(result: dict[str, Any]) -> None:
    paths = result.get("paths", {})
    if not paths:
        st.caption("No saved outputs are available yet.")
        return

    st.caption(f"Run output folder: {result.get('run_dir')}")
    with st.container(horizontal=True):
        for key, path_text in paths.items():
            path = Path(path_text)
            data = read_bytes_if_exists(path)
            if not data:
                continue
            mime = "application/json" if path.suffix == ".json" else "application/jsonl"
            st.download_button(
                f"Download {key}",
                data=data,
                file_name=path.name,
                mime=mime,
                icon=":material/download:",
            )
        if result.get("cards"):
            st.download_button(
                "Download manual labeling CSV",
                data=cards_to_manual_csv_bytes(result["cards"]),
                file_name="conflict_cards_manual_labels.csv",
                mime="text/csv",
                icon=":material/rate_review:",
            )


with st.sidebar:
    st.header("Run settings")
    if not SEMANTIC_SCHOLAR_API_KEY:
        st.warning(
            "Semantic Scholar key is missing. The app will try unauthenticated access.",
            icon=":material/info:",
        )
    if not GROQ_API_KEY:
        st.warning(
            "Groq key is missing. Groq validation and explanations will fall back or be skipped.",
            icon=":material/info:",
        )

    with st.form("pipeline_settings"):
        query = st.text_input(
            "Research topic",
            value="retrieval augmented generation hallucination reduction",
            placeholder="Enter a literature review topic",
        )
        paper_limit = st.slider("Paper limit", min_value=10, max_value=50, value=30, step=5)
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.30,
            max_value=0.80,
            value=0.45,
            step=0.05,
        )
        contradiction_threshold = st.slider(
            "Contradiction threshold",
            min_value=0.50,
            max_value=0.90,
            value=0.65,
            step=0.05,
        )
        max_cards = st.slider("Max cards", min_value=5, max_value=30, value=20, step=5)
        use_cache = st.checkbox("Use cached paper retrieval", value=True)
        use_groq = st.checkbox("Use Groq", value=True)
        recover_missing_claims = st.checkbox("Recover up to 3 key claims per abstract", value=True)
        submitted = st.form_submit_button(
            "Run pipeline",
            type="primary",
            icon=":material/play_arrow:",
        )


st.title("Contradiction Finder for Research Reading")
st.caption(
    "Enter a topic, retrieve abstracts, extract normalized claims, pair related cross-paper claims, "
    "score possible contradictions locally, and generate cautious evidence-grounded conflict cards."
)

if submitted:
    with st.status("Running pipeline...", expanded=True) as status:
        try:
            result = run_contradiction_pipeline(
                query=query,
                paper_limit=paper_limit,
                use_cache=use_cache,
                use_groq=use_groq,
                recover_missing_claims=recover_missing_claims,
                similarity_threshold=similarity_threshold,
                contradiction_threshold=contradiction_threshold,
                max_cards=max_cards,
            )
            st.session_state["last_result"] = result
            status.update(label="Pipeline complete", state="complete")
        except Exception as exc:
            status.update(label="Pipeline stopped", state="error")
            st.error(str(exc), icon=":material/error:")

result = st.session_state.get("last_result")
if result:
    for warning in result.get("warnings", []):
        st.warning(warning, icon=":material/warning:")

    render_metrics(result)
    view = st.segmented_control(
        "View",
        ["Conflict cards", "Retrieved papers", "Extracted claims", "Candidate pairs", "Exports"],
        default="Conflict cards",
    )

    if view == "Retrieved papers":
        render_papers(result["papers"])
    elif view == "Extracted claims":
        render_claims(result["claims"], result["papers"])
    elif view == "Candidate pairs":
        render_pairs(result["pairs"], result["claims"], result["papers"], result["scores"])
    elif view == "Exports":
        render_exports(result)
    else:
        render_conflict_cards(result["cards"])
else:
    with st.container(border=True):
        st.markdown("**Ready to inspect a literature topic**")
        st.write(
            "Use the sidebar controls to retrieve abstracts and build a transparent set of "
            "possible contradiction cards. The app saves papers, sentences, claims, pairs, "
            "scores, cards, and run stats as query-specific JSONL files."
        )
