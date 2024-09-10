# On the reproduction of methods in Mammoth

The following table contains the list of models implemented in Mammoth and the state of 
reproducibility of the results in the original papers. 

| Model                     | `--model`                | Reproduces? | WandB run | Notes |
| ------------------------- | ------------------------ | ----------- | --------- | ----- |
| A-GEM R                   | `agem_r`                 | X           |           |       |
| A-GEM                     | `agem`                   | X           |           |       |
| AttriClip                 | `attriclip`              | X           |           |       |
| BiC                       | `bic`                    | X           |           |       |
| CCIC                      | `ccic`                   | X           |           |       |
| CGIL                      | `cgil`                   | X           |           |       |
| CLIP                      | `clip`                   | X           |           |       |
| Coda Prompt               | `coda_prompt`            | X           |           |       |
| DAP                       | `dap`                    | X           |           |       |
| DER                       | `der`                    | X           |           |       |
| DER++ LiDER               | `derpp_lider`            | X           |           |       |
| DER++                     | `derpp`                  | X           |           |       |
| DualPrompt                | `dualprompt`             | X           |           |       |
| ER-ACE LiDER              | `er_ace_aer_abs`         | X           |           |       |
| ER-ACE with AER and ABS   | `er_ace_lider`           | X           |           |       |
| ER-ACE with Tricks        | `er_ace_tricks`          | X           |           |       |
| ER-ACE                    | `er_ace`                 | X           |           |       |
| ER                        | `er_tricks`              | X           |           |       |
| EwC Online                | `er`                     | X           |           |       |
| FDR                       | `ewc_on`                 | X           |           |       |
| First-Stage STAR-Prompt   | `fdr`                    | X           |           |       |
| GdumB LiDER               | `first_stage_starprompt` | X           |           |       |
| GdumB                     | `gdumb_lider`            | X           |           |       |
| GEM                       | `gdumb`                  | X           |           |       |
| GSS                       | `gem`                    | X           |           |       |
| HAL                       | `gss`                    | X           |           |       |
| iCaRL LiDER               | `hal`                    | X           |           |       |
| iCaRL                     | `icarl_lider`            | X           |           |       |
| Joint GCL                 | `icarl`                  | X           |           |       |
| Joint                     | `joint_gcl`              | X           |           |       |
| L2P                       | `joint`                  | X           |           |       |
| LLAVA                     | `l2p`                    | X           |           |       |
| LUCIR                     | `llava`                  | X           |           |       |
| LwF-MC                    | `lucir`                  | X           |           |       |
| LwF                       | `lwf_mc`                 | X           |           |       |
| MER                       | `lwf`                    | X           |           |       |
| MoE Adapters              | `mer`                    | X           |           |       |
| PNN                       | `moe_adapters`           | X           |           |       |
| Puridiver                 | `pnn`                    | X           |           |       |
| Ranpac                    | `puridiver`              | X           |           |       |
| RPC                       | `ranpac`                 | X           |           |       |
| Second-Stage STAR-Prompt  | `rpc`                    | X           |           |       |
| SGD                       | `second_stage_starprompt`| X           |           |       |
| SI                        | `sgd`                    | X           |           |       |
| SLCA                      | `si`                     | X           |           |       |
| STAP-Prompt               | `slca`                   | X           |           |       |
| TwF                       | `starprompt`             | X           |           |       |
| X-DER with CE             | `twf`                    | X           |           |       |
| X-DER with RPC            | `xder_ce`                | X           |           |       |
| X-DER                     | `xder_rpc`               | X           |           |       |