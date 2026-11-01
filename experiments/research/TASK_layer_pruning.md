# Research task for agy (literature survey, do NOT modify repo code)

Context: we will depth-prune the LTX-2.5 22B audio-video DiT (joint video+audio streams, flow matching,
teacher = ltx-2.5-22b-dev) to a student keeping ~50% of transformer blocks, initialized from teacher weights,
then recover via offline knowledge distillation (teacher CFG-baked velocity targets + optional feature distillation).
Goal: inference speedup with minimal quality loss; no quantization. Hardware: 4x H200.

Question: what is the best method to choose WHICH blocks to keep/remove? Search the web and survey:
1. Block importance metrics for DiT / video DiT depth pruning (e.g. TinyFusion learnable depth pruning, BK-SDM,
   ShortGPT Block Influence / input-output cosine similarity, Taylor/gradient importance, Flux.1 Lite (Freepik),
   compression work on HunyuanVideo / Wan / CogVideoX / SANA / SnapGen, anything LTX-related).
2. Whether importance should be measured per timestep range.
3. Contiguous vs uniform vs metric-based removal; empirical comparisons.
4. Block merging / averaging alternatives to plain removal.
5. Recovery training recipes: losses, feature-distillation layer mapping, data size, steps, LR, reported compute.
6. Special considerations for joint audio-video blocks with cross-modal attention.

For each method: paper title, year, link, key numbers, cost.
End with a concrete recommended procedure for our case, including a cheap calibration experiment to rank blocks.

Write the full report in Traditional Chinese to experiments/research/layer_pruning_survey.md, then reply DONE.
