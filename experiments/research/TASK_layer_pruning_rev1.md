# Revision request for layer_pruning_survey.md (from supervisor)

Do not rewrite from scratch; produce experiments/research/layer_pruning_survey_rev1.md.

1. Our teacher has changed: it is the OFFICIAL ltx-2.5-22b-DISTILLED transformer (8 fixed sigmas stage 1 + 4 stage 2, no CFG).
   Drop the CFG-baking content; adapt timestep weighting to the 8+4 fixed sigma schedule.
2. Mark every claim as [paper-verified] (give exact section/table) or [unverified/heuristic]. In particular check:
   - "first/last 2 blocks must never be removed", "max contiguous skip <= 3", the 15/65/20 timestep weights,
     "VBench >= 96%", "AV sync < 40ms", "0.6-0.8 s/step", "13 A100-days" for BK-SDM, Flux.1 Lite 23% / 7GB.
   Remove or label any number you cannot source.
3. TinyFusion reports that one-shot calibration-loss importance correlates poorly with post-finetune quality
   (recoverability). Verify and quote what they found, and summarise any other evidence about
   zero-shot metric vs. after-training quality.
4. The JAVIS formula adds unbounded gate norms to bounded cosine scores. Either justify with a source or drop it.
5. Find concrete evaluation tooling for audio-video generation: FVD implementation, FAD, CLAP score,
   AV-sync metrics (SyncNet / Synchformer DeSync / ImageBind AV), VBench dims. Give repo links.

Reply DONE when the file is written.
