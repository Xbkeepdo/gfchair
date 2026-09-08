# Counterfactual failure and recovery history

- Image `120777`: the first formal run exhausted a 24 GiB GPU while caching complete 32-layer captures for several pixel-counterfactual rectangles.
- Recovery keeps only the four required layers' counterfactual visual `h_prev` states, current target rows, and logits, then immediately releases each complete capture. Completed image shards were retained and the failed image is retried by `--resume`.
- Formal acceptance still requires all 100 selected images, 400 target-layer cases, complete three-family/three-strategy mappings, finite values, checksums, and exactly 10,000 bootstrap resamples.
