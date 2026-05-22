# Knowledge Base Index

This folder is the local RAG source for the app.

Location:
- `knowledge_base/chunks/` contains the knowledge chunks used at runtime.

Chunk format:
- Markdown files.
- Optional YAML-style frontmatter with `title`, `tags`, and `summary`.
- Keep each chunk short, focused, and self-contained.

How the app uses the KB:
- Step 1 queries the EDA selection logic chunk.
- Step 2 queries data cleaning techniques and modelling constraints.
- Step 3 queries modelling selection and adaptation usage chunks.
- Step 4 queries model evaluations chunk.
- Every step also queries the system adaptation protocol chunk first.

Recommended chunk layout:
- One topic per file.
- Prefer 180-350 words per chunk for free-tier efficiency.
- Put high-signal rules first; examples second.
- Use clear tags like `routing`, `modeling`, `cleaning`, `eda`, `validation`.

Efficiency targets:
- Default retrieval top-k should be step-aware:
	- Step 1: 2
	- Step 2: 2
	- Step 3: 3
	- Step 4: 2
- Keep chunk summaries short and specific to increase ranking precision.

Current seed chunks:
- `modeling_routing.md`
- `rag_testing_note.md`

Operational chunks:
- `system_adaptation_protocol.md`
- `system_adaptation_usage.md`
- `eda_selection_logic.md`
- `data_cleaning_techniques.md`
- `data_modelling_selection.md`
- `model_evaluations.md`