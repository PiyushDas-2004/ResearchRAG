# ADR 0004: Parent-child (small-to-big) chunking as the default

- Status: accepted
- Date: 2026-11-21

## Context
Small chunks retrieve precisely but give the model too little context; large chunks dilute
embeddings and retrieval precision. Citations must point to pages, so chunk offsets must map back
to the extracted text exactly.

## Decision
- All chunkers operate on spans of the parsed text (`chunk.text == doc.text[start:end]`), with
  page numbers and a per-chunk page map for exact citation pages.
- Default `parent_child`: section-aware parents (~480 tokens, never crossing a section boundary)
  are sent to the model; sentence-packed children (~160 tokens, 32-token overlap) are embedded and
  retrieved. Tables are kept atomic when they fit.
- `fixed`, `sentence` and `section` strategies are available and swept in evaluation. Chunk sets
  are keyed by the chunking config hash, so strategies can coexist in one database.
- An optional contextual header (paper title + section) is prepended to the text that is
  embedded (not to the stored chunk).
- References are chunked and stored (for metadata) but excluded from retrieval by default.

## Consequences
- Property-based tests enforce coverage, token limits and overlap bounds.
- Parent passages cost more context tokens; the packer enforces a token budget.
