"""Convert paper abstracts into traceable sentence records."""

from __future__ import annotations

from src.schemas import Paper, PaperSentence
from src.utils.hashing import safe_paper_id
from src.utils.text import clean_text, split_into_sentences, word_count


def split_papers_into_sentences(papers: list[Paper]) -> list[PaperSentence]:
    sentences: list[PaperSentence] = []
    for paper in papers:
        safe_id = safe_paper_id(paper.paper_id)
        for idx, sentence in enumerate(split_into_sentences(paper.abstract)):
            sentence = clean_text(sentence)
            if word_count(sentence) < 8:
                continue
            sentences.append(
                PaperSentence(
                    sentence_id=f"{safe_id}_S{idx}",
                    paper_id=paper.paper_id,
                    sentence_index=idx,
                    text=sentence,
                )
            )
    return sentences

