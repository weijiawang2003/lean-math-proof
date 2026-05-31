# TR1 model card

Pilot failure-to-action router. NOT production routing. Trained on verified
proof-search artifacts (see tr1_methodology.md).

- examples: 57, labels: 7
- best model: sgd (macro-F1 0.628)
- beats rule baseline: True
- intended use: prioritize a next-work queue (SF5 retrieval / deeper search).
- limitations: tiny corpus, several singleton classes, name features may dominate.
- do NOT use for production routing or to credit any candidate.