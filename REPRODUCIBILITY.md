# On the reproduction of methods in Mammoth

The following table contains the list of models implemented in Mammoth and the state of 
reproducibility of the results in the original papers. 

| Model                     | `--model`                | Reproduces? | WandB run | Notes |
| ------------------------- | ------------------------ | ----------- | --------- | ----- |
| A-GEM R                   | `agem_r`                 | X           |           |       |
| A-GEM                     | `agem`                   | X           |           |       |
| AttriClip                 | `attriclip`              | X           |           | The original repo was pulled because it did not reproduce. |
| BiC                       | `bic`                    | X           |           |       |
| CCIC                      | `ccic`                   | V           |           |       |
| CGIL                      | `cgil`                   | V           |           |       |
| CLIP                      | `clip`                   | V           |           |       |
| CODA-Prompt               | `coda_prompt`            | V           |           |       |
| DAP                       | `dap`                    | V           |           |       |
| DER                       | `der`                    | V           |           |       |
| DER++ LiDER               | `derpp_lider`            | V           |           |       |
| DER++                     | `derpp`                  | V           |           |       |
| DualPrompt                | `dualprompt`             | V           |           |       |
| ER-ACE LiDER              | `er_ace_aer_abs`         | X           |           |       |
| ER-ACE with AER and ABS   | `er_ace_lider`           | X           |           |       |
| ER-ACE with Tricks        | `er_ace_tricks`          | X           |           |       |
| ER-ACE                    | `er_ace`                 | X           |           |       |
| EwC Online                | `er`                     | X           |           |       |
| ER                        | `ewc_on`                 | X           |           |       |
| FDR                       | `fdr`                    | X           |           |       |
| First-Stage STAR-Prompt   | `first_stage_starprompt` | X           |           |       |
| GdumB LiDER               | `gdumb_lider`            | V           |           |       |
| GdumB                     | `gdumb`                  | V           |           |       |
| GEM                       | `gem`                    | V           |           | Original work requires too much resources. We reproduced the results in DER. |
| GSS                       | `gss`                    | X           |           |       |
| HAL                       | `hal`                    | X           |           |       |
| iCaRL LiDER               | `icarl_lider`            | X           |           |       |
| iCaRL                     | `icarl`                  | X           |           |       |
| Joint GCL                 | `joint_gcl`              | X           |           |       |
| Joint                     | `joint`                  | X           |           |       |
| L2P                       | `l2p`                    | V           |           |       |
| LLAVA                     | `llava`                  | X           |           |       |
| LUCIR                     | `lucir`                  | X           |           |       |
| LwF-MC                    | `lwf_mc`                 | X           |           |       |
| LwF                       | `lwf`                    | X           |           |       |
| MER                       | `mer`                    | X           |           |       |
| MoE Adapters              | `moe_adapters`           | V           |           |       |
| PNN                       | `pnn`                    | X           |           |       |
| Puridiver                 | `puridiver`              | X           |           |       |
| Ranpac                    | `ranpac`                 | X           |           |       |
| RPC                       | `rpc`                    | X           |           |       |
| Second-Stage STAR-Prompt  | `second_stage_starprompt`| X           |           |       |
| SGD                       | `sgd`                    | X           |           |       |
| SI                        | `si`                     | X           |           |       |
| SLCA                      | `slca`                   | X           |           |       |
| STAP-Prompt               | `starprompt`             | X           |           |       |
| TwF                       | `twf`                    | X           |           |       |
| X-DER with CE             | `xder_ce`                | X           |           |       |
| X-DER with RPC            | `xder_rpc`               | X           |           |       |
| X-DER                     | `xder`                   | X           |           |       |