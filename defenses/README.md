# Defense integrations

The first defense adapter will run RAGDefender after retrieval and before
documents are passed to a generator:

```text
query -> retriever -> retrieved passages -> RAGDefender -> safe passages -> LLM
```

RAGDefender is kept as an external, version-pinned dependency rather than
copying or modifying its upstream source in this repository.
