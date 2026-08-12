# LangGraph flow

The Q&A path is intentionally small and inspectable:

```text
                         +------------------+
                         |                  v
START -> retrieve -> assess -- relevant -> answer -> END
                         |
                         +-- insufficient -> fallback -> END
```

## Nodes

1. **retrieve** calls the Pinecone adapter with the user's question and gets the top `TOP_K` chunks, including their source metadata and similarity scores.
2. **assess** is a deterministic retrieval gate. It requires a configurable minimum top score and minimum overlap between meaningful question terms and the retrieved evidence. This avoids sending unrelated chunks to the model.
3. **answer** receives only the retrieved chunks. Its prompt forbids outside knowledge and requires chunk IDs after material claims. The API's citation objects are copied directly from those retrieved Pinecone records.
4. **fallback** is the bad-evidence path. It emits no citations and says the information cannot be found in the supplied corpus.

## Bounds and grounding

The state contains `step_count` and `max_steps`. Nodes increment the counter and refuse to continue once the configured bound is reached. The service also passes `recursion_limit = MAX_GRAPH_STEPS + 2` to LangGraph. There is currently no retry loop: a weak retrieval result takes the fallback branch immediately, which makes the endpoint predictable and prevents repeated LLM calls.

## State

The relevant state fields are `question`, `chunks`, `retrieval_ok`, `answer`, `grounded`, `citations`, `trace`, `step_count`, and `max_steps`. Trace entries identify the node, route, match count, score, and lexical overlap without including secrets.

