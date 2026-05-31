# SF5 training delta summary

- examples exported (additive): **20**
- label histogram: {'PROOF_DEPTH_GAP': 15, 'RETRIEVAL_ROUTING_GAP': 1, 'EXISTING_LEMMA_GAP': 4}
- skipped guard classes: {}

Does NOT modify TR1/TR2 datasets. 

## How TR3/TR4 should use these

Merge additively with tr1_examples.jsonl as a retrieval-aware label channel; positive labels (EXISTING_LEMMA_GAP / RETRIEVAL_ROUTING_GAP) train a router to fire library-search / lemma-retrieval actions; negatives keep verified-label discipline.
