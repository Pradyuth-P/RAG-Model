# RAG Tuning: Chunk Size and Overlap Optimization Guidelines

Optimizing retrieval-augmented generation (RAG) performance requires careful tuning of chunk size and chunk overlap parameters. Selecting inappropriate parameters can result in poor retrieval relevance or lost context.

## 1. Chunk Size Considerations

- **Small Chunks (100 - 500 characters)**:
  - Pros: High precision, lower retrieval costs, fewer token footprints.
  - Cons: Context fragmentation. The model may miss the overall thesis or core message.
- **Medium Chunks (500 - 1500 characters) - *Recommended***:
  - Pros: Preserves local context while keeping search highly relevant. Fits well in LLM context windows.
  - Cons: Minor redundancy in overlapping sections.
- **Large Chunks (1500 - 4000 characters)**:
  - Pros: Retains full document sections and continuity.
  - Cons: Retrieval contains noise. Higher token costs, slower generation times.

## 2. Chunk Overlap Guidelines

Chunk overlap prevents information from being bisected at the boundary of a chunk.
- **Standard Overlap**: 10% to 20% of the chunk size. For a 1000-character chunk, use a 200-character overlap.
- **High-OverLap (25%+)**: Useful when dealing with highly dense technical formulas or numerical records where adjacent sentences are tightly coupled.

## 3. Evaluative Metrics for Retrieval

When tuning chunk parameters, measure performance using:
1. **Context Recall**: Can the retriever fetch all parts of the document necessary to answer the question?
2. **Context Precision**: Are all the retrieved chunks relevant, or do they contain fluff?
3. **Faithfulness / Groundedness**: Does the LLM answer rest *solely* on the context, or does it hallucinate?

## 4. Default Setup for AetherRAG

The default settings for AetherRAG are:
- `chunk_size = 1000`
- `chunk_overlap = 200`
- Splitter: `RecursiveCharacterTextSplitter` (which splits sequentially on `\n\n`, `\n`, ` `, and `""`).
