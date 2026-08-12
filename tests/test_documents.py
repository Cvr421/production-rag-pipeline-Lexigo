from lexigo.documents import load_markdown_documents, split_markdown


def test_split_markdown_preserves_sections_and_stable_ids() -> None:
    text = (
        "# Title\n\n## Facts\n\nThe facts are fictional.\n\n"
        "## Remedy\n\nThe remedy is stated here."
    )

    first = split_markdown(text, "example.md")
    second = split_markdown(text, "example.md")

    assert len(first) == 3
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[1].source_file == "example.md"
    assert "Facts" in first[1].text


def test_sample_corpus_loads() -> None:
    chunks = load_markdown_documents("gen_ai_takehome_sample_corpus")

    assert chunks
    assert {chunk.source_file for chunk in chunks} == {
        "01_matter_memo_arvind_v_northfield.md",
        "02_employment_agreement_excerpt.md",
        "03_hearing_notice_template.md",
        "04_statute_style_excerpt_fictional.md",
        "05_counsel_notes_settlement.md",
        "06_property_lease_clause.md",
    }
