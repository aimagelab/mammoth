# On the reproduction of methods in Mammoth

The following table contains a list of models implemented in Mammoth and a checklist to indicate if we already have a reproduction of the results in the original papers.

| Model                     | `--model`                | Verified? | Notes |
| ------------------------- | ------------------------ | --------- | ----- |
| A-GEM R                   | `agem_r`                 | X         |       |
| A-GEM                     | `agem`                   | X         |       |
| AttriClip                 | `attriclip`              | X         | The original repo was pulled because it did not reproduce. |
| BiC                       | `bic`                    | X         |       |
| CCIC                      | `ccic`                   | V         |       |
| CGIL                      | `cgil`                   | V         |       |
| CLIP                      | `clip`                   | V         |       |
| CNLL                      | `cnll`                   | V         |       |
| CODA-Prompt               | `coda_prompt`            | V         |       |
| DAP                       | `dap`                    | V         |       |
| DER                       | `der`                    | V         |       |
| DER++                     | `derpp`                  | V         |       |
| DER++ LiDER               | `derpp_lider`            | V         |       |
| DER++ & STAR              | `derpp_star`             | V         |       |
| DualPrompt                | `dualprompt`             | V         |       |
| ER-ACE                    | `er_ace`                 | V         |       |
| ER-ACE LiDER              | `er_ace_aer_abs`         | V         |       |
| ER-ACE with AER and ABS   | `er_ace_lider`           | V         |       |
| ER-ACE with Tricks        | `er_ace_tricks`          | N/A       | This method does not come from a paper. |
| ER-ACE & STAR             | `er_ace_star`            | V         |       |
| EwC Online                | `ewc_on`                 | X         |       |
| ER                        | `er`                     | N/A       | This method predates modern benchmarks (R. Ratcliff. "Connectionist models of recognition memory: constraints imposed by learning and forgetting functions", 1990 -- A. Robins, "Catastrophic forgetting, rehearsal and pseudorehearsal", 1995) |
| ER & STAR                 | `er_star`                | V         |       |
| FDR                       | `fdr`                    | X         |       |
| First-Stage STAR-Prompt   | `first_stage_starprompt` | V         |       |
| GdumB LiDER               | `gdumb_lider`            | V         |       |
| GdumB                     | `gdumb`                  | V         |       |
| GEM                       | `gem`                    | V         | Original work requires too much resources. We reproduced the results in `Dark Experience for General Continual Learning: a Strong, Simple Baseline`. |
| GSS                       | `gss`                    | X         |       |
| HAL                       | `hal`                    | X         |       |
| iCaRL LiDER               | `icarl_lider`            | V         |       |
| iCaRL                     | `icarl`                  | V         |       |
| IEL                       | `iel`                    | V         | The model is `second_order` with `use_iel=1`. |
| ITA                       | `ita`                    | V         | The model is `second_order` with `use_iel=0`. |
| Joint GCL                 | `joint_gcl`              | N/A       | There is no single paper for the "joint" model. |
| Joint                     | `joint`                  | N/A       | There is no single paper for the "joint" model. |
| L2P                       | `l2p`                    | V         |       |
| LLAVA                     | `llava`                  | N/A       | This method does not come from a CL paper. |
| LUCIR                     | `lucir`                  | X         |       |
| LwF-MC                    | `lwf_mc`                 | X         |       |
| LwF                       | `lwf`                    | X         |       |
| LwS                       | `lws`                    | V         |       |
| MER                       | `mer`                    | X         |       |
| MoE Adapters              | `moe_adapters`           | V         |       |
| PNN                       | `pnn`                    | X         |       |
| Puridiver                 | `puridiver`              | X         |       |
| Ranpac                    | `ranpac`                 | X         |       |
| RPC                       | `rpc`                    | X         |       |
| Second-Stage STAR-Prompt  | `second_stage_starprompt`| V         |       |
| SGD                       | `sgd`                    | N/A       | There is no single paper for the "SGD" model. |
| SI                        | `si`                     | X         |       |
| SLCA                      | `slca`                   | X         |       |
| SPR                       | `spr`                    | V         | Training takes around 3 1/2 days on a 1080ti for `seq-mnist` |
| STAR-Prompt               | `starprompt`             | V         |       |
| TwF                       | `twf`                    | V         |       |
| X-DER                     | `xder`                   | V         |       |
| X-DER with CE             | `xder_ce`                | V         |       |
| X-DER with RPC            | `xder_rpc`               | V         |       |
| X-DER with RPC & STAR     | `xder_rpc_star`          | V         |       |