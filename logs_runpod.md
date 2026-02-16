root@30844596adcb:/workspace/Finetuning_LLama# python scripts/15_train_dpo.py
================================================================================
DPO TRAINING — Stage 3: Preference Optimization with LoRA
================================================================================
Start time: 2026-02-16 11:42:17
Hardware: NVIDIA RTX 6000 Ada Generation
VRAM: 47.5 GB
================================================================================
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info

1. Loading merged DAPT+SFT model from: ./models/merged_sft/
The tokenizer you are loading from './models/merged_sft/' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:23<00:00,  5.87s/it]
The module name  (originally ) is not a valid Python identifier. Please rename the original module to avoid import issues.

   Tokenizer Details:
     Class: PreTrainedTokenizerFast
     Vocab size: 128256
     Model max length: 131072
     BOS: <|begin_of_text|> (id: 128000)
     EOS: <|eot_id|> (id: 128009)
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
     PAD: <|pad|> (id: 128256) [ADDED]
   ✓ Model loaded: 8,030,269,440 parameters

2. Loading reference model (frozen copy for DPO)...
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.53it/s]
The module name  (originally ) is not a valid Python identifier. Please rename the original module to avoid import issues.
   Resizing reference model embeddings: 128256 → 128257
   ✓ Reference model loaded and frozen

3. Applying DPO LoRA adapter to policy model...
   ✓ DPO LoRA applied to policy model
trainable params: 41,943,040 || all params: 8,072,212,480 || trainable%: 0.5196

4. Loading DPO dataset...
   ✓ Train: 2016 pairs
   ✓ Eval: 231 pairs

5. Setting up DPO training configuration...
   ✓ DPO Config:
     Beta: 0.1
     Loss type: sigmoid
     Epochs: 3
     Batch size: 1
     Grad accumulation: 16
     Effective batch: 16
     Learning rate: 5e-06
     Label smoothing: 0.05
     LoRA rank: 16
     Max length: 1024
     Early stopping: patience=3 evals on eval_loss
     Health guard: stop if margins > 5.0 or loss < 0.05

6. Initializing DPO Trainer...
The model is already on multiple devices. Skipping the move to device specified in `args`.
   ✓ Trainer initialized
     Total training steps: ~378

7. Starting DPO training...
================================================================================
Training in progress... Monitor with:
  tensorboard --logdir=logs/dpo
================================================================================
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 128009, 'pad_token_id': 128256}.
{'loss': 0.6931, 'grad_norm': 5.802131652832031, 'learning_rate': 0.0, 'rewards/chosen': 0.0, 'rewards/rejected': 0.0, 'rewards/accuracies': 0.0, 'rewards/margins': 0.0, 'logps/chosen': -268.03912353515625, 'logps/rejected': -1212.9830322265625, 'logits/chosen': -0.8508821725845337, 'logits/rejected': -0.7725358605384827, 'epoch': 0.01}
{'loss': 0.6956, 'grad_norm': 5.134267807006836, 'learning_rate': 1.1842105263157894e-06, 'rewards/chosen': 0.003373716026544571, 'rewards/rejected': 0.004422166384756565, 'rewards/accuracies': 0.4375, 'rewards/margins': -0.0010484507074579597, 'logps/chosen': -278.4651184082031, 'logps/rejected': -1121.968505859375, 'logits/chosen': -0.8254848718643188, 'logits/rejected': -0.8037469387054443, 'epoch': 0.08}
{'loss': 0.6912, 'grad_norm': 5.352558612823486, 'learning_rate': 2.5e-06, 'rewards/chosen': 0.0017132090870290995, 'rewards/rejected': -0.006560773588716984, 'rewards/accuracies': 0.48750001192092896, 'rewards/margins': 0.008273984305560589, 'logps/chosen': -281.83447265625, 'logps/rejected': -1108.966552734375, 'logits/chosen': -0.877062201499939, 'logits/rejected': -0.8335579037666321, 'epoch': 0.16}
{'loss': 0.685, 'grad_norm': 6.3880438804626465, 'learning_rate': 3.815789473684211e-06, 'rewards/chosen': -0.008023958653211594, 'rewards/rejected': -0.03054114244878292, 'rewards/accuracies': 0.574999988079071, 'rewards/margins': 0.022517183795571327, 'logps/chosen': -284.5879211425781, 'logps/rejected': -1158.88916015625, 'logits/chosen': -0.8755080103874207, 'logits/rejected': -0.8188766241073608, 'epoch': 0.24}
{'loss': 0.6283, 'grad_norm': 5.711842060089111, 'learning_rate': 4.99989327925842e-06, 'rewards/chosen': -0.016888286918401718, 'rewards/rejected': -0.172427698969841, 'rewards/accuracies': 0.887499988079071, 'rewards/margins': 0.155539408326149, 'logps/chosen': -271.1617126464844, 'logps/rejected': -1123.965087890625, 'logits/chosen': -0.8500906825065613, 'logits/rejected': -0.8180688619613647, 'epoch': 0.32}
{'loss': 0.5179, 'grad_norm': 5.440282344818115, 'learning_rate': 4.987097811457015e-06, 'rewards/chosen': -0.05317042022943497, 'rewards/rejected': -0.5098029375076294, 'rewards/accuracies': 0.9937499761581421, 'rewards/margins': 0.45663246512413025, 'logps/chosen': -293.1388244628906, 'logps/rejected': -1117.4600830078125, 'logits/chosen': -0.830925464630127, 'logits/rejected': -0.8396786451339722, 'epoch': 0.4}
{'eval_loss': 0.4417378902435303, 'eval_runtime': 148.0671, 'eval_samples_per_second': 1.56, 'eval_steps_per_second': 1.56, 'eval_rewards/chosen': -0.07467612624168396, 'eval_rewards/rejected': -0.7779611945152283, 'eval_rewards/accuracies': 0.9956709742546082, 'eval_rewards/margins': 0.7032850384712219, 'eval_logps/chosen': -276.1766357421875, 'eval_logps/rejected': -1126.0977783203125, 'eval_logits/chosen': -0.838113009929657, 'eval_logits/rejected': -0.8327344059944153, 'epoch': 0.4}
 13%|██████████████████▉                                                                                                                            | 50/378 [20:03<1:52:59, 20.67s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.3612, 'grad_norm': 3.059044122695923, 'learning_rate': 4.953083302054247e-06, 'rewards/chosen': -0.1133660301566124, 'rewards/rejected': -1.1799724102020264, 'rewards/accuracies': 1.0, 'rewards/margins': 1.0666064023971558, 'logps/chosen': -291.7447509765625, 'logps/rejected': -1170.633056640625, 'logits/chosen': -0.8539546728134155, 'logits/rejected': -0.8238593339920044, 'epoch': 0.48}
{'loss': 0.2441, 'grad_norm': 0.24963705241680145, 'learning_rate': 4.898139950851163e-06, 'rewards/chosen': -0.22712311148643494, 'rewards/rejected': -2.342784881591797, 'rewards/accuracies': 1.0, 'rewards/margins': 2.115661144256592, 'logps/chosen': -290.6859130859375, 'logps/rejected': -1134.074462890625, 'logits/chosen': -0.8691008687019348, 'logits/rejected': -0.8412524461746216, 'epoch': 0.56}
{'loss': 0.2237, 'grad_norm': 0.6283015012741089, 'learning_rate': 4.822736515147748e-06, 'rewards/chosen': -0.3554694354534149, 'rewards/rejected': -3.7293567657470703, 'rewards/accuracies': 1.0, 'rewards/margins': 3.373886823654175, 'logps/chosen': -273.7433776855469, 'logps/rejected': -1154.1343994140625, 'logits/chosen': -0.8665668368339539, 'logits/rejected': -0.8722019195556641, 'epoch': 0.63}
{'loss': 0.2169, 'grad_norm': 0.4080509543418884, 'learning_rate': 4.72751631047092e-06, 'rewards/chosen': -0.3276122212409973, 'rewards/rejected': -3.6757705211639404, 'rewards/accuracies': 1.0, 'rewards/margins': 3.348158359527588, 'logps/chosen': -275.35150146484375, 'logps/rejected': -1174.14599609375, 'logits/chosen': -0.8573335409164429, 'logits/rejected': -0.848837673664093, 'epoch': 0.71}
{'loss': 0.2098, 'grad_norm': 0.2950875759124756, 'learning_rate': 4.613291722033285e-06, 'rewards/chosen': -0.26919639110565186, 'rewards/rejected': -3.113593578338623, 'rewards/accuracies': 1.0, 'rewards/margins': 2.8443973064422607, 'logps/chosen': -290.58416748046875, 'logps/rejected': -1160.1016845703125, 'logits/chosen': -0.8905377388000488, 'logits/rejected': -0.8395413160324097, 'epoch': 0.79}
{'eval_loss': 0.21084590256214142, 'eval_runtime': 148.826, 'eval_samples_per_second': 1.552, 'eval_steps_per_second': 1.552, 'eval_rewards/chosen': -0.24387666583061218, 'eval_rewards/rejected': -2.967360496520996, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.7234833240509033, 'eval_logps/chosen': -277.8686218261719, 'eval_logps/rejected': -1147.99169921875, 'eval_logits/chosen': -0.850165605545044, 'eval_logits/rejected': -0.8478521704673767, 'epoch': 0.79}
 26%|█████████████████████████████████████▌                                                                                                        | 100/378 [40:33<1:40:43, 21.74s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2096, 'grad_norm': 0.1413063108921051, 'learning_rate': 4.481037273748935e-06, 'rewards/chosen': -0.2481682300567627, 'rewards/rejected': -3.041980266571045, 'rewards/accuracies': 1.0, 'rewards/margins': 2.7938122749328613, 'logps/chosen': -277.5808410644531, 'logps/rejected': -1166.056884765625, 'logits/chosen': -0.916434645652771, 'logits/rejected': -0.861890435218811, 'epoch': 0.87}
{'loss': 0.2123, 'grad_norm': 0.1982187032699585, 'learning_rate': 4.3318813139390295e-06, 'rewards/chosen': -0.2716798186302185, 'rewards/rejected': -3.2011985778808594, 'rewards/accuracies': 1.0, 'rewards/margins': 2.929518938064575, 'logps/chosen': -285.0367126464844, 'logps/rejected': -1166.8619384765625, 'logits/chosen': -0.8649082183837891, 'logits/rejected': -0.8606348037719727, 'epoch': 0.95}
{'loss': 0.2103, 'grad_norm': 0.22686435282230377, 'learning_rate': 4.167096388661754e-06, 'rewards/chosen': -0.2459661215543747, 'rewards/rejected': -3.2645275592803955, 'rewards/accuracies': 1.0, 'rewards/margins': 3.018561363220215, 'logps/chosen': -277.538330078125, 'logps/rejected': -1156.007080078125, 'logits/chosen': -0.8656994700431824, 'logits/rejected': -0.8627824783325195, 'epoch': 1.03}
{'loss': 0.2055, 'grad_norm': 0.26355642080307007, 'learning_rate': 3.9880883847980475e-06, 'rewards/chosen': -0.24812161922454834, 'rewards/rejected': -3.145143508911133, 'rewards/accuracies': 1.0, 'rewards/margins': 2.897022008895874, 'logps/chosen': -288.547607421875, 'logps/rejected': -1153.2584228515625, 'logits/chosen': -0.8760564923286438, 'logits/rejected': -0.8409024477005005, 'epoch': 1.11}
{'loss': 0.2095, 'grad_norm': 0.37893715500831604, 'learning_rate': 3.7963845355204303e-06, 'rewards/chosen': -0.22895283997058868, 'rewards/rejected': -3.1149237155914307, 'rewards/accuracies': 1.0, 'rewards/margins': 2.8859705924987793, 'logps/chosen': -283.1956481933594, 'logps/rejected': -1161.2733154296875, 'logits/chosen': -0.8656278848648071, 'logits/rejected': -0.8604402542114258, 'epoch': 1.19}
{'eval_loss': 0.2060304582118988, 'eval_runtime': 147.7796, 'eval_samples_per_second': 1.563, 'eval_steps_per_second': 1.563, 'eval_rewards/chosen': -0.22042956948280334, 'eval_rewards/rejected': -3.1789889335632324, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.958559513092041, 'eval_logps/chosen': -277.6341552734375, 'eval_logps/rejected': -1150.1082763671875, 'eval_logits/chosen': -0.8493445515632629, 'eval_logits/rejected': -0.8465741276741028, 'epoch': 1.19}
 40%|███████████████████████████████████████████████████████▌                                                                                    | 150/378 [1:00:47<1:19:58, 21.05s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2037, 'grad_norm': 0.2365809679031372, 'learning_rate': 3.5936203904780665e-06, 'rewards/chosen': -0.24281954765319824, 'rewards/rejected': -3.256472110748291, 'rewards/accuracies': 1.0, 'rewards/margins': 3.013652801513672, 'logps/chosen': -282.0604553222656, 'logps/rejected': -1193.4306640625, 'logits/chosen': -0.8892942667007446, 'logits/rejected': -0.8349849581718445, 'epoch': 1.27}
{'loss': 0.2057, 'grad_norm': 0.08748380839824677, 'learning_rate': 3.3815258618638316e-06, 'rewards/chosen': -0.21536056697368622, 'rewards/rejected': -3.148752212524414, 'rewards/accuracies': 1.0, 'rewards/margins': 2.93339204788208, 'logps/chosen': -284.3255615234375, 'logps/rejected': -1152.5474853515625, 'logits/chosen': -0.8873801231384277, 'logits/rejected': -0.8647617101669312, 'epoch': 1.35}
{'loss': 0.2025, 'grad_norm': 0.4053947925567627, 'learning_rate': 3.1619104654134397e-06, 'rewards/chosen': -0.22325126826763153, 'rewards/rejected': -3.1490869522094727, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9258358478546143, 'logps/chosen': -289.75518798828125, 'logps/rejected': -1165.3692626953125, 'logits/chosen': -0.8810027241706848, 'logits/rejected': -0.8367830514907837, 'epoch': 1.43}
{'loss': 0.2046, 'grad_norm': 0.17265291512012482, 'learning_rate': 2.9366478822551973e-06, 'rewards/chosen': -0.21036167442798615, 'rewards/rejected': -3.1785521507263184, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9681906700134277, 'logps/chosen': -291.51727294921875, 'logps/rejected': -1159.766845703125, 'logits/chosen': -0.8439338803291321, 'logits/rejected': -0.8346317410469055, 'epoch': 1.51}
{'loss': 0.2021, 'grad_norm': 0.16051281988620758, 'learning_rate': 2.70765997332326e-06, 'rewards/chosen': -0.1920800507068634, 'rewards/rejected': -3.1408424377441406, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9487626552581787, 'logps/chosen': -283.1018981933594, 'logps/rejected': -1180.97119140625, 'logits/chosen': -0.8399246335029602, 'logits/rejected': -0.8169872164726257, 'epoch': 1.59}
{'eval_loss': 0.20417390763759613, 'eval_runtime': 147.3962, 'eval_samples_per_second': 1.567, 'eval_steps_per_second': 1.567, 'eval_rewards/chosen': -0.1917564421892166, 'eval_rewards/rejected': -3.1593759059906006, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.9676196575164795, 'eval_logps/chosen': -277.3474426269531, 'eval_logps/rejected': -1149.912109375, 'eval_logits/chosen': -0.8483105897903442, 'eval_logits/rejected': -0.8455821871757507, 'epoch': 1.59}
 53%|██████████████████████████████████████████████████████████████████████████                                                                  | 200/378 [1:21:05<1:07:03, 22.60s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.207, 'grad_norm': 0.3111278712749481, 'learning_rate': 2.4769003827177735e-06, 'rewards/chosen': -0.21470603346824646, 'rewards/rejected': -3.0950284004211426, 'rewards/accuracies': 1.0, 'rewards/margins': 2.880322217941284, 'logps/chosen': -276.244873046875, 'logps/rejected': -1118.3756103515625, 'logits/chosen': -0.8820589184761047, 'logits/rejected': -0.8802653551101685, 'epoch': 1.67}
{'loss': 0.2049, 'grad_norm': 0.15733368694782257, 'learning_rate': 2.246337869902243e-06, 'rewards/chosen': -0.20389166474342346, 'rewards/rejected': -3.1660897731781006, 'rewards/accuracies': 1.0, 'rewards/margins': 2.962198257446289, 'logps/chosen': -283.2813415527344, 'logps/rejected': -1145.76806640625, 'logits/chosen': -0.8590779304504395, 'logits/rejected': -0.8173822164535522, 'epoch': 1.75}
{'loss': 0.2041, 'grad_norm': 0.16690368950366974, 'learning_rate': 2.017939512941958e-06, 'rewards/chosen': -0.18330471217632294, 'rewards/rejected': -3.1838877201080322, 'rewards/accuracies': 1.0, 'rewards/margins': 3.0005831718444824, 'logps/chosen': -283.7197570800781, 'logps/rejected': -1170.84375, 'logits/chosen': -0.8419599533081055, 'logits/rejected': -0.859337329864502, 'epoch': 1.83}
{'loss': 0.2031, 'grad_norm': 0.1622404307126999, 'learning_rate': 1.7936539260875125e-06, 'rewards/chosen': -0.19349969923496246, 'rewards/rejected': -3.1252148151397705, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9317150115966797, 'logps/chosen': -281.06439208984375, 'logps/rejected': -1158.23486328125, 'logits/chosen': -0.8876329660415649, 'logits/rejected': -0.8223863840103149, 'epoch': 1.9}
{'loss': 0.2031, 'grad_norm': 0.1685837209224701, 'learning_rate': 1.5753946348850774e-06, 'rewards/chosen': -0.18909728527069092, 'rewards/rejected': -3.125667095184326, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9365696907043457, 'logps/chosen': -287.54595947265625, 'logps/rejected': -1171.951904296875, 'logits/chosen': -0.898189902305603, 'logits/rejected': -0.859115481376648, 'epoch': 1.98}
{'eval_loss': 0.20272715389728546, 'eval_runtime': 148.0824, 'eval_samples_per_second': 1.56, 'eval_steps_per_second': 1.56, 'eval_rewards/chosen': -0.1841178983449936, 'eval_rewards/rejected': -3.1456072330474854, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.961489200592041, 'eval_logps/chosen': -277.27105712890625, 'eval_logps/rejected': -1149.774169921875, 'eval_logits/chosen': -0.8474414348602295, 'eval_logits/rejected': -0.8446716070175171, 'epoch': 1.98}
 66%|█████████████████████████████████████████████████████████████████████████████████████████████▉                                                | 250/378 [1:41:15<45:54, 21.52s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2016, 'grad_norm': 0.20451822876930237, 'learning_rate': 1.3650237506511333e-06, 'rewards/chosen': -0.18362528085708618, 'rewards/rejected': -3.1364896297454834, 'rewards/accuracies': 1.0, 'rewards/margins': 2.952864170074463, 'logps/chosen': -272.4907531738281, 'logps/rejected': -1159.723876953125, 'logits/chosen': -0.8520804643630981, 'logits/rejected': -0.833290696144104, 'epoch': 2.06}
{'loss': 0.2035, 'grad_norm': 0.2175413966178894, 'learning_rate': 1.164336083595279e-06, 'rewards/chosen': -0.18033930659294128, 'rewards/rejected': -3.1201412677764893, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9398016929626465, 'logps/chosen': -282.1368103027344, 'logps/rejected': -1144.824462890625, 'logits/chosen': -0.8426012992858887, 'logits/rejected': -0.8405278325080872, 'epoch': 2.14}
{'loss': 0.2037, 'grad_norm': 0.2365809679031372, 'learning_rate': 3.5936203904780665e-06, 'rewards/chosen': -0.24281954765319824, 'rewards/rejected': -3.256472110748291, 'rewards/accuracies': 1.0, 'rewards/margins': 3.013652801513672, 'logps/chosen': -282.0604553222656, 'logps/rejected': -1193.4306640625, 'logits/chosen': -0.8892942667007446, 'logits/rejected': -0.8349849581718445, 'epoch': 1.27}
{'loss': 0.2057, 'grad_norm': 0.08748380839824677, 'learning_rate': 3.3815258618638316e-06, 'rewards/chosen': -0.21536056697368622, 'rewards/rejected': -3.148752212524414, 'rewards/accuracies': 1.0, 'rewards/margins': 2.93339204788208, 'logps/chosen': -284.3255615234375, 'logps/rejected': -1152.5474853515625, 'logits/chosen': -0.8873801231384277, 'logits/rejected': -0.8647617101669312, 'epoch': 1.35}
{'loss': 0.2025, 'grad_norm': 0.4053947925567627, 'learning_rate': 3.1619104654134397e-06, 'rewards/chosen': -0.22325126826763153, 'rewards/rejected': -3.1490869522094727, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9258358478546143, 'logps/chosen': -289.75518798828125, 'logps/rejected': -1165.3692626953125, 'logits/chosen': -0.8810027241706848, 'logits/rejected': -0.8367830514907837, 'epoch': 1.43}
{'loss': 0.2046, 'grad_norm': 0.17265291512012482, 'learning_rate': 2.9366478822551973e-06, 'rewards/chosen': -0.21036167442798615, 'rewards/rejected': -3.1785521507263184, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9681906700134277, 'logps/chosen': -291.51727294921875, 'logps/rejected': -1159.766845703125, 'logits/chosen': -0.8439338803291321, 'logits/rejected': -0.8346317410469055, 'epoch': 1.51}
{'loss': 0.2021, 'grad_norm': 0.16051281988620758, 'learning_rate': 2.70765997332326e-06, 'rewards/chosen': -0.1920800507068634, 'rewards/rejected': -3.1408424377441406, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9487626552581787, 'logps/chosen': -283.1018981933594, 'logps/rejected': -1180.97119140625, 'logits/chosen': -0.8399246335029602, 'logits/rejected': -0.8169872164726257, 'epoch': 1.59}
{'eval_loss': 0.20417390763759613, 'eval_runtime': 147.3962, 'eval_samples_per_second': 1.567, 'eval_steps_per_second': 1.567, 'eval_rewards/chosen': -0.1917564421892166, 'eval_rewards/rejected': -3.1593759059906006, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.9676196575164795, 'eval_logps/chosen': -277.3474426269531, 'eval_logps/rejected': -1149.912109375, 'eval_logits/chosen': -0.8483105897903442, 'eval_logits/rejected': -0.8455821871757507, 'epoch': 1.59}
 53%|██████████████████████████████████████████████████████████████████████████                                                                  | 200/378 [1:21:05<1:07:03, 22.60s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.207, 'grad_norm': 0.3111278712749481, 'learning_rate': 2.4769003827177735e-06, 'rewards/chosen': -0.21470603346824646, 'rewards/rejected': -3.0950284004211426, 'rewards/accuracies': 1.0, 'rewards/margins': 2.880322217941284, 'logps/chosen': -276.244873046875, 'logps/rejected': -1118.3756103515625, 'logits/chosen': -0.8820589184761047, 'logits/rejected': -0.8802653551101685, 'epoch': 1.67}
{'loss': 0.2049, 'grad_norm': 0.15733368694782257, 'learning_rate': 2.246337869902243e-06, 'rewards/chosen': -0.20389166474342346, 'rewards/rejected': -3.1660897731781006, 'rewards/accuracies': 1.0, 'rewards/margins': 2.962198257446289, 'logps/chosen': -283.2813415527344, 'logps/rejected': -1145.76806640625, 'logits/chosen': -0.8590779304504395, 'logits/rejected': -0.8173822164535522, 'epoch': 1.75}
{'loss': 0.2041, 'grad_norm': 0.16690368950366974, 'learning_rate': 2.017939512941958e-06, 'rewards/chosen': -0.18330471217632294, 'rewards/rejected': -3.1838877201080322, 'rewards/accuracies': 1.0, 'rewards/margins': 3.0005831718444824, 'logps/chosen': -283.7197570800781, 'logps/rejected': -1170.84375, 'logits/chosen': -0.8419599533081055, 'logits/rejected': -0.859337329864502, 'epoch': 1.83}
{'loss': 0.2031, 'grad_norm': 0.1622404307126999, 'learning_rate': 1.7936539260875125e-06, 'rewards/chosen': -0.19349969923496246, 'rewards/rejected': -3.1252148151397705, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9317150115966797, 'logps/chosen': -281.06439208984375, 'logps/rejected': -1158.23486328125, 'logits/chosen': -0.8876329660415649, 'logits/rejected': -0.8223863840103149, 'epoch': 1.9}
{'loss': 0.2031, 'grad_norm': 0.1685837209224701, 'learning_rate': 1.5753946348850774e-06, 'rewards/chosen': -0.18909728527069092, 'rewards/rejected': -3.125667095184326, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9365696907043457, 'logps/chosen': -287.54595947265625, 'logps/rejected': -1171.951904296875, 'logits/chosen': -0.898189902305603, 'logits/rejected': -0.859115481376648, 'epoch': 1.98}
{'eval_loss': 0.20272715389728546, 'eval_runtime': 148.0824, 'eval_samples_per_second': 1.56, 'eval_steps_per_second': 1.56, 'eval_rewards/chosen': -0.1841178983449936, 'eval_rewards/rejected': -3.1456072330474854, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.961489200592041, 'eval_logps/chosen': -277.27105712890625, 'eval_logps/rejected': -1149.774169921875, 'eval_logits/chosen': -0.8474414348602295, 'eval_logits/rejected': -0.8446716070175171, 'epoch': 1.98}
 66%|█████████████████████████████████████████████████████████████████████████████████████████████▉                                                | 250/378 [1:41:15<45:54, 21.52s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2016, 'grad_norm': 0.20451822876930237, 'learning_rate': 1.3650237506511333e-06, 'rewards/chosen': -0.18362528085708618, 'rewards/rejected': -3.1364896297454834, 'rewards/accuracies': 1.0, 'rewards/margins': 2.952864170074463, 'logps/chosen': -272.4907531738281, 'logps/rejected': -1159.723876953125, 'logits/chosen': -0.8520804643630981, 'logits/rejected': -0.833290696144104, 'epoch': 2.06}
{'loss': 0.2035, 'grad_norm': 0.2175413966178894, 'learning_rate': 1.164336083595279e-06, 'rewards/chosen': -0.18033930659294128, 'rewards/rejected': -3.1201412677764893, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9398016929626465, 'logps/chosen': -282.1368103027344, 'logps/rejected': -1144.824462890625, 'logits/chosen': -0.8426012992858887, 'logits/rejected': -0.8405278325080872, 'epoch': 2.14}
{'loss': 0.2031, 'grad_norm': 0.1927879899740219, 'learning_rate': 9.750438301323584e-07, 'rewards/chosen': -0.1713300496339798, 'rewards/rejected': -3.1192195415496826, 'rewards/accuracies': 1.0, 'rewards/margins': 2.947889804840088, 'logps/chosen': -280.6810607910156, 'logps/rejected': -1135.5147705078125, 'logits/chosen': -0.8895753622055054, 'logits/rejected': -0.8510559797286987, 'epoch': 2.22}
{'loss': 0.2021, 'grad_norm': 0.17333315312862396, 'learning_rate': 7.987619650263462e-07, 'rewards/chosen': -0.19327959418296814, 'rewards/rejected': -3.141083240509033, 'rewards/accuracies': 1.0, 'rewards/margins': 2.947803497314453, 'logps/chosen': -293.5706481933594, 'logps/rejected': -1156.40576171875, 'logits/chosen': -0.902661144733429, 'logits/rejected': -0.8350445628166199, 'epoch': 2.3}
{'loss': 0.2012, 'grad_norm': 0.15634380280971527, 'learning_rate': 6.369944629950933e-07, 'rewards/chosen': -0.18354913592338562, 'rewards/rejected': -3.168691635131836, 'rewards/accuracies': 1.0, 'rewards/margins': 2.985142469406128, 'logps/chosen': -286.26361083984375, 'logps/rejected': -1197.3153076171875, 'logits/chosen': -0.8635610342025757, 'logits/rejected': -0.8553042411804199, 'epoch': 2.38}
{'eval_loss': 0.20249667763710022, 'eval_runtime': 148.1203, 'eval_samples_per_second': 1.56, 'eval_steps_per_second': 1.56, 'eval_rewards/chosen': -0.16891714930534363, 'eval_rewards/rejected': -3.119828939437866, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.9509119987487793, 'eval_logps/chosen': -277.1190185546875, 'eval_logps/rejected': -1149.5164794921875, 'eval_logits/chosen': -0.8471508026123047, 'eval_logits/rejected': -0.8440437912940979, 'epoch': 2.38}
 79%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                             | 300/378 [2:01:41<28:47, 22.14s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2022, 'grad_norm': 0.19249077141284943, 'learning_rate': 4.911214673283157e-07, 'rewards/chosen': -0.1718759834766388, 'rewards/rejected': -3.134277820587158, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9624016284942627, 'logps/chosen': -287.7836608886719, 'logps/rejected': -1184.868896484375, 'logits/chosen': -0.8516471982002258, 'logits/rejected': -0.8327493667602539, 'epoch': 2.46}
{'loss': 0.2028, 'grad_norm': 0.20607203245162964, 'learning_rate': 3.623875149916725e-07, 'rewards/chosen': -0.1629531979560852, 'rewards/rejected': -3.080014705657959, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9170613288879395, 'logps/chosen': -279.0074768066406, 'logps/rejected': -1144.1173095703125, 'logits/chosen': -0.8612211346626282, 'logits/rejected': -0.8268709182739258, 'epoch': 2.54}
{'loss': 0.2015, 'grad_norm': 0.14103655517101288, 'learning_rate': 2.5189091867618615e-07, 'rewards/chosen': -0.17596520483493805, 'rewards/rejected': -3.0772345066070557, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9012691974639893, 'logps/chosen': -290.4861145019531, 'logps/rejected': -1172.162109375, 'logits/chosen': -0.8651034235954285, 'logits/rejected': -0.8310586810112, 'epoch': 2.62}
{'loss': 0.2018, 'grad_norm': 0.21574446558952332, 'learning_rate': 1.6057439638161891e-07, 'rewards/chosen': -0.16808956861495972, 'rewards/rejected': -3.1226840019226074, 'rewards/accuracies': 1.0, 'rewards/margins': 2.954594135284424, 'logps/chosen': -273.76129150390625, 'logps/rejected': -1145.0067138671875, 'logits/chosen': -0.8817472457885742, 'logits/rejected': -0.8577890396118164, 'epoch': 2.7}
{'loss': 0.2028, 'grad_norm': 0.2042781263589859, 'learning_rate': 8.921702847888791e-08, 'rewards/chosen': -0.16436012089252472, 'rewards/rejected': -3.111262321472168, 'rewards/accuracies': 1.0, 'rewards/margins': 2.94690203666687, 'logps/chosen': -293.48681640625, 'logps/rejected': -1150.703369140625, 'logits/chosen': -0.8419862985610962, 'logits/rejected': -0.8427208662033081, 'epoch': 2.78}
{'eval_loss': 0.20222586393356323, 'eval_runtime': 148.0559, 'eval_samples_per_second': 1.56, 'eval_steps_per_second': 1.56, 'eval_rewards/chosen': -0.16721440851688385, 'eval_rewards/rejected': -3.128024101257324, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.9608099460601807, 'eval_logps/chosen': -277.10198974609375, 'eval_logps/rejected': -1149.5985107421875, 'eval_logits/chosen': -0.8468148112297058, 'eval_logits/rejected': -0.8438452482223511, 'epoch': 2.78}
 93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍          | 350/378 [2:21:58<09:54, 21.24s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2022, 'grad_norm': 0.12722158432006836, 'learning_rate': 3.8427610871041024e-08, 'rewards/chosen': -0.1811288744211197, 'rewards/rejected': -3.1579408645629883, 'rewards/accuracies': 1.0, 'rewards/margins': 2.976811647415161, 'logps/chosen': -286.52142333984375, 'logps/rejected': -1199.709716796875, 'logits/chosen': -0.8781858682632446, 'logits/rejected': -0.8212734460830688, 'epoch': 2.86}
{'loss': 0.2017, 'grad_norm': 0.2128118872642517, 'learning_rate': 8.639460961263612e-09, 'rewards/chosen': -0.16412286460399628, 'rewards/rejected': -3.1410412788391113, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9769184589385986, 'logps/chosen': -278.73504638671875, 'logps/rejected': -1164.9061279296875, 'logits/chosen': -0.8711044192314148, 'logits/rejected': -0.8630857467651367, 'epoch': 2.94}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 378/378 [2:31:47<00:00, 20.51s/it]/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'train_runtime': 9117.7233, 'train_samples_per_second': 0.663, 'train_steps_per_second': 0.041, 'train_loss': 0.26855763714149516, 'epoch': 3.0}                                      
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 378/378 [2:31:57<00:00, 24.12s/it]

8. Saving DPO LoRA adapters...
/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(

================================================================================
DPO TRAINING COMPLETE
================================================================================
End time: 2026-02-16 14:14:57
DPO LoRA adapters saved to: checkpoints/dpo_lora/final
Logs: logs/dpo

Training pipeline complete:
  ✓ Stage 1: DAPT LoRA (domain adaptation)
  ✓ Stage 2: SFT LoRA (instruction tuning)
  ✓ Stage 3: DPO LoRA (preference optimization)

Next step: Merge DPO LoRA into final model
  python scripts/16_merge_dpo_lora.py
================================================================================
root@30844596adcb:/workspace/Finetuning_LLama# 


root@30844596adcb:/workspace/Finetuning_LLama# python diagnose_dpo_quality.py
================================================================================
DIAGNOSING DPO TRAINING QUALITY
================================================================================

1. Loading SFT model...
The tokenizer you are loading from './models/merged_sft' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
`torch_dtype` is deprecated! Use `dtype` instead!
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:22<00:00,  5.62s/it]
   ✓ SFT model loaded

2. Loading DPO model...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.57it/s]
The module name  (originally ) is not a valid Python identifier. Please rename the original module to avoid import issues.
Traceback (most recent call last):
  File "/workspace/Finetuning_LLama/diagnose_dpo_quality.py", line 183, in <module>
    results = test_reward_hacking(sft_model_path, dpo_model_path, test_samples=10)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Finetuning_LLama/diagnose_dpo_quality.py", line 79, in test_reward_hacking
    dpo_model, dpo_tokenizer = load_model(dpo_model_path)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Finetuning_LLama/diagnose_dpo_quality.py", line 24, in load_model
    model = AutoModelForCausalLM.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py", line 604, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py", line 277, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py", line 5148, in from_pretrained
    model.load_adapter(
  File "/usr/local/lib/python3.11/dist-packages/transformers/integrations/peft.py", line 261, in load_adapter
    incompatible_keys = set_peft_model_state_dict(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py", line 565, in set_peft_model_state_dict
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 2629, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for LlamaForCausalLM:
        size mismatch for model.embed_tokens.weight: copying a param with shape torch.Size([128257, 4096]) from checkpoint, the shape in current model is torch.Size([128256, 4096]).
        size mismatch for lm_head.weight: copying a param with shape torch.Size([128257, 4096]) from checkpoint, the shape in current model is torch.Size([128256, 4096]).
root@30844596adcb:/workspace/Finetuning_LLama# python v
validate_dpo_dataset.py   validate_dpo_training.py  verify_tokenizer.py       
root@30844596adcb:/workspace/Finetuning_LLama# python validate_dpo_training.py 
================================================================================
VALIDATING DPO TRAINING RESULTS
================================================================================

1. Loading SFT model (baseline)...
Loading model from ./models/merged_sft...
The tokenizer you are loading from './models/merged_sft' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
`torch_dtype` is deprecated! Use `dtype` instead!
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.56it/s]
   ✓ SFT model loaded

2. Generating responses from SFT model...
   Generating 1/5...
   Generating 2/5...
   Generating 3/5...
   Generating 4/5...
   Generating 5/5...

3. Loading DPO model...
Loading model from checkpoints/dpo_lora/final...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.50it/s]
The module name  (originally ) is not a valid Python identifier. Please rename the original module to avoid import issues.
Traceback (most recent call last):
  File "/workspace/Finetuning_LLama/validate_dpo_training.py", line 183, in <module>
    validate_training()
  File "/workspace/Finetuning_LLama/validate_dpo_training.py", line 95, in validate_training
    dpo_model, dpo_tokenizer = load_model(str(dpo_checkpoint))
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Finetuning_LLama/validate_dpo_training.py", line 25, in load_model
    model = AutoModelForCausalLM.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py", line 604, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py", line 277, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py", line 5148, in from_pretrained
    model.load_adapter(
  File "/usr/local/lib/python3.11/dist-packages/transformers/integrations/peft.py", line 261, in load_adapter
    incompatible_keys = set_peft_model_state_dict(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py", line 565, in set_peft_model_state_dict
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 2629, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for LlamaForCausalLM:
        size mismatch for model.embed_tokens.weight: copying a param with shape torch.Size([128257, 4096]) from checkpoint, the shape in current model is torch.Size([128256, 4096]).
        size mismatch for lm_head.weight: copying a param with shape torch.Size([128257, 4096]) from checkpoint, the shape in current model is torch.Size([128256, 4096]).
root@30844596adcb:/workspace/Finetuning_LLama# 



root@30844596adcb:/workspace/Finetuning_LLama# python scripts/16_merge_dpo_lora.py
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
================================================================================
MERGE DPO LORA → FINAL PRODUCTION MODEL
================================================================================
Start time: 2026-02-16 14:35:19
Base (merged SFT): ./models/merged_sft/
DPO LoRA: ./checkpoints/dpo_lora/final/
Output: ./models/final_dpo/
================================================================================

1. Loading merged DAPT+SFT model...
The tokenizer you are loading from 'models/merged_sft' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:21<00:00,  5.26s/it]
   ✓ Model loaded: 8,030,261,248 parameters

1b. Checking tokenizer compatibility...
   Resizing model embeddings: 128256 → 128257
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
   ✓ Embeddings resized to match DPO checkpoint

2. Merging DPO LoRA...
   ✓ DPO LoRA merged

3. Saving final production model to ./models/final_dpo/...

================================================================================
FINAL MODEL MERGE COMPLETE
================================================================================
End time: 2026-02-16 14:36:31
Final model: models/final_dpo
Model size: 14.96 GB

This model includes ALL 3 training stages:
  ✓ Stage 1: DAPT LoRA (KP domain adaptation) — merged
  ✓ Stage 2: SFT LoRA (instruction tuning) — merged
  ✓ Stage 3: DPO LoRA (preference optimization) — merged

Next step: Quantize for deployment
  python scripts/06_quantize_unsloth.py --model models/final_dpo
================================================================================
root@30844596adcb:/workspace/Finetuning_LLama# python diagnose_dpo_quality.py
================================================================================
DIAGNOSING DPO TRAINING QUALITY
================================================================================

1. Loading SFT model...
The tokenizer you are loading from './models/merged_sft' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
`torch_dtype` is deprecated! Use `dtype` instead!
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:21<00:00,  5.43s/it]
   ✓ SFT model loaded

2. Loading DPO model...
The tokenizer you are loading from './models/final_dpo' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:22<00:00,  5.50s/it]
   ✓ DPO model loaded

3. Loading test pairs...
   ✓ Loaded 231 test pairs

4. Testing 10 samples...
================================================================================

### Sample 1 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

DPO Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: +0.00
  ⚠️  UNCLEAR: Mixed signals

### Sample 2 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   -67.00 (avg: -2.6719)
  Rejected logp: -1520.00 (avg: -2.8125)
  Margin: 1453.00

DPO Model:
  Chosen logp:   -66.50 (avg: -2.6562)
  Rejected logp: -1536.00 (avg: -2.8281)
  Margin: 1469.50

Changes (DPO - SFT):
  Chosen:   +0.50
  Rejected: -16.00
  ⚠️  WARNING: Rejected collapsed more than chosen improved!
  This suggests reward hacking (annihilating rejected)

### Sample 3 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

DPO Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: +0.00
  ⚠️  UNCLEAR: Mixed signals

### Sample 4 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

DPO Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: +0.00
  ⚠️  UNCLEAR: Mixed signals

### Sample 5 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

DPO Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: +0.00
  ⚠️  UNCLEAR: Mixed signals

### Sample 6 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

DPO Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: +0.00
  ⚠️  UNCLEAR: Mixed signals

### Sample 7 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

DPO Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: +0.00
  ⚠️  UNCLEAR: Mixed signals

### Sample 8 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

DPO Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: +0.00
  ⚠️  UNCLEAR: Mixed signals

### Sample 9 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

DPO Model:
  Chosen logp:   0.00 (avg: nan)
  Rejected logp: 0.00 (avg: nan)
  Margin: 0.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: +0.00
  ⚠️  UNCLEAR: Mixed signals

### Sample 10 ###
Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Jyotish, a warm and confident K...

SFT Model:
  Chosen logp:   -498.00 (avg: -2.5156)
  Rejected logp: -1512.00 (avg: -2.4844)
  Margin: 1014.00

DPO Model:
  Chosen logp:   -498.00 (avg: -2.5156)
  Rejected logp: -1528.00 (avg: -2.5000)
  Margin: 1030.00

Changes (DPO - SFT):
  Chosen:   +0.00
  Rejected: -16.00
  ⚠️  WARNING: Rejected collapsed more than chosen improved!
  This suggests reward hacking (annihilating rejected)

================================================================================
SUMMARY
================================================================================

Average changes (DPO - SFT):
  Chosen:   +0.05
  Rejected: -3.20
  Ratio: 64.00x

Reward hacking detected: 2/10 samples

⚠️  WARNING: Some samples show reward hacking.
Model may have mixed quality.
root@30844596adcb:/workspace/Finetuning_LLama# python validate_dpo_training.py
================================================================================
VALIDATING DPO TRAINING RESULTS
================================================================================

1. Loading SFT model (baseline)...
Loading model from ./models/merged_sft...
The tokenizer you are loading from './models/merged_sft' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
`torch_dtype` is deprecated! Use `dtype` instead!
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.44it/s]
   ✓ SFT model loaded

2. Generating responses from SFT model...
   Generating 1/5...
   Generating 2/5...
   Generating 3/5...
   Generating 4/5...
   Generating 5/5...

3. Loading DPO model...
Loading model from models/final_dpo...
The tokenizer you are loading from 'models/final_dpo' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.39it/s]
   ✓ DPO model loaded

4. Generating responses from DPO model...
   Generating 1/5...
   Generating 2/5...
   Generating 3/5...
   Generating 4/5...
   Generating 5/5...

================================================================================
COMPARISON RESULTS
================================================================================

### Test 1: When will I get married?
--------------------------------------------------------------------------------

SFT Response (949 chars):
   How can I find my life partner? What should I look for in a marriage partner?
These are fundamental questions that have puzzled many seekers of truth throughout history. In KP astrology, marriage is ...

DPO Response (796 chars):
   How can I find a good life partner? What should I look for in a marriage partner?
Marriage timing depends on your dasha periods and the ruling planets at the time of your query. For marriage, you nee...

Analysis:
  Length ratio (DPO/SFT): 0.84x
  ✅ Length is reasonable
  ✅ No excessive repetition

### Test 2: Will I get a job promotion this year?
--------------------------------------------------------------------------------

SFT Response (777 chars):
   How do I analyze my career prospects using houses 2, 3, 6, and 11?
In KP astrology, analyzing job promotion requires examining houses 2, 3, 6, and 11, as these bhavas specifically indicate career adv...

DPO Response (858 chars):
   How can I use KP astrology to determine my career prospects and timing for advancement?
To determine if you'll receive a job promotion this year using KP astrology, focus on the sub-lord of the 10th ...

Analysis:
  Length ratio (DPO/SFT): 1.10x
  ✅ Length is reasonable
  ✅ No excessive repetition

### Test 3: What is my career prospect?
--------------------------------------------------------------------------------

SFT Response (984 chars):
   How will I succeed in profession? Which profession will suit me? How can I grow in my career?
Your career prospects are excellent based on the strong planetary combinations in your chart. The 9th hou...

DPO Response (949 chars):
   How can I use this KP rule to identify favorable periods for starting a business?
To determine your career prospects using KP astrology, analyze the 10th cusp sub-lord's placement and connections. Th...

Analysis:
  Length ratio (DPO/SFT): 0.96x
  ✅ Length is reasonable
  ✅ No excessive repetition

### Test 4: When will I have children?
--------------------------------------------------------------------------------

SFT Response (937 chars):
   How many children will I have? These are fundamental questions that many people seek answers to. In KP astrology, the 5th house is the primary significator for children and progeny matters.

The timi...

DPO Response (855 chars):
   How many will I have?
In KP astrology, the timing and number of children are determined through a specific combination of houses and their ruling planets. According to KP principles, the 5th house go...

Analysis:
  Length ratio (DPO/SFT): 0.91x
  ✅ Length is reasonable
  ✅ No excessive repetition

### Test 5: Should I invest in real estate?
--------------------------------------------------------------------------------

SFT Response (1243 chars):
   Here's a comprehensive guide to help you make an informed decision about real estate investments.
Real estate investment is a significant financial decision that requires careful analysis of various ...

DPO Response (1200 chars):
   My financial advisor says now isn't the right time due to market volatility.
When considering real estate investment during market volatility, it's crucial to understand that timing can be favorable ...

Analysis:
  Length ratio (DPO/SFT): 0.97x
  ✅ Length is reasonable
  ✅ No excessive repetition

================================================================================
SUMMARY
================================================================================

Average length ratio (DPO/SFT): 0.96x

✅ GOOD: Response lengths are reasonable.
DPO training appears successful.

================================================================================
RECOMMENDATION
================================================================================

✅ Model appears ready for deployment.
Proceed with merging and testing.
root@30844596adcb:/workspace/Finetuning_LLama# 


root@30844596adcb:/workspace/Finetuning_LLama# python scripts/08_serve_vllm.py   --max-model-len 128000
================================================================================
vLLM INFERENCE SERVER — KP Astrology Model
================================================================================
  Model:      models/merged
  Server:     http://0.0.0.0:8000/v1
  Max length: 128000
  Dtype:      auto
  APC:        ON
  KV cache:   auto
  GPU memory: 90%
================================================================================

Endpoints:
  POST http://0.0.0.0:8000/v1/chat/completions
  POST http://0.0.0.0:8000/v1/completions
  GET  http://0.0.0.0:8000/health

Example curl:
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "models/merged", "messages": [{"role": "user", "content": "What is the 7th house sub-lord significance in KP astrology?"}]}'

Press Ctrl+C to stop.
================================================================================
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
(APIServer pid=10887) INFO 02-16 14:52:15 [utils.py:325] 
(APIServer pid=10887) INFO 02-16 14:52:15 [utils.py:325]        █     █     █▄   ▄█
(APIServer pid=10887) INFO 02-16 14:52:15 [utils.py:325]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.15.1
(APIServer pid=10887) INFO 02-16 14:52:15 [utils.py:325]   █▄█▀ █     █     █     █  model   models/merged
(APIServer pid=10887) INFO 02-16 14:52:15 [utils.py:325]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=10887) INFO 02-16 14:52:15 [utils.py:325] 
(APIServer pid=10887) INFO 02-16 14:52:15 [utils.py:261] non-default args: {'host': '0.0.0.0', 'model': 'models/merged', 'trust_remote_code': True, 'max_model_len': 128000, 'served_model_name': ['kp-astrology-llama'], 'enable_prefix_caching': True}
(APIServer pid=10887) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=10887) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=10887) INFO 02-16 14:52:23 [model.py:541] Resolved architecture: LlamaForCausalLM
(APIServer pid=10887) INFO 02-16 14:52:23 [model.py:1561] Using max model len 128000
(APIServer pid=10887) INFO 02-16 14:52:24 [scheduler.py:226] Chunked prefill is enabled with max_num_batched_tokens=2048.
(APIServer pid=10887) INFO 02-16 14:52:24 [vllm.py:624] Asynchronous scheduling is enabled.
(APIServer pid=10887) The tokenizer you are loading from 'models/merged' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
(EngineCore_DP0 pid=11280) INFO 02-16 14:52:30 [core.py:96] Initializing a V1 LLM engine (v0.15.1) with config: model='models/merged', speculative_config=None, tokenizer='models/merged', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=128000, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=kp-astrology-llama, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': True, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': True}, 'local_cache_dir': None, 'static_all_moe_layers': []}
(EngineCore_DP0 pid=11280) INFO 02-16 14:52:31 [parallel_state.py:1212] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.19.0.2:34233 backend=nccl
(EngineCore_DP0 pid=11280) INFO 02-16 14:52:31 [parallel_state.py:1423] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A
(EngineCore_DP0 pid=11280) INFO 02-16 14:52:31 [gpu_model_runner.py:4033] Starting to load model models/merged...
(EngineCore_DP0 pid=11280) INFO 02-16 14:52:47 [cuda.py:364] Using FLASH_ATTN attention backend out of potential backends: ('FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION')
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:01<00:04,  1.63s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:08<00:09,  4.76s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:15<00:05,  5.89s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:23<00:00,  6.41s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:23<00:00,  5.76s/it]
(EngineCore_DP0 pid=11280) 
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:10 [default_loader.py:291] Loading weights took 23.17 seconds
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:11 [gpu_model_runner.py:4130] Model loading took 14.99 GiB memory and 38.680632 seconds
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:14 [backends.py:812] Using cache directory: /root/.cache/vllm/torch_compile_cache/cdede397e9/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:14 [backends.py:872] Dynamo bytecode transform time: 3.45 s
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:21 [backends.py:302] Cache the graph of compile range (1, 2048) for later use
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:25 [backends.py:319] Compiling a graph for compile range (1, 2048) takes 6.40 s
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:25 [monitor.py:34] torch.compile takes 9.85 s in total
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:26 [gpu_worker.py:356] Available KV cache memory: 26.54 GiB
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:26 [kv_cache_utils.py:1307] GPU KV cache size: 217,376 tokens
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:26 [kv_cache_utils.py:1312] Maximum concurrency for 128,000 tokens per request: 1.70x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|███████████████████████████████████████████████████████████████████████| 51/51 [00:02<00:00, 18.99it/s]
Capturing CUDA graphs (decode, FULL): 100%|██████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:01<00:00, 22.66it/s]
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:31 [gpu_model_runner.py:5063] Graph capturing finished in 5 secs, took 0.55 GiB
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:31 [core.py:272] init engine (profile, create kv cache, warmup model) took 20.24 seconds
(EngineCore_DP0 pid=11280) The tokenizer you are loading from 'models/merged' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
(EngineCore_DP0 pid=11280) INFO 02-16 14:53:32 [vllm.py:624] Asynchronous scheduling is enabled.
(APIServer pid=10887) INFO 02-16 14:53:32 [api_server.py:665] Supported tasks: ['generate']
(APIServer pid=10887) WARNING 02-16 14:53:32 [model.py:1371] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.6, 'top_p': 0.9}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=10887) INFO 02-16 14:53:32 [serving.py:177] Warming up chat template processing...
(APIServer pid=10887) The tokenizer you are loading from 'models/merged' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
(APIServer pid=10887) INFO 02-16 14:53:32 [hf.py:310] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=10887) INFO 02-16 14:53:32 [serving.py:212] Chat template warmup completed in 402.3ms
(APIServer pid=10887) INFO 02-16 14:53:32 [api_server.py:946] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:38] Available routes are:
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /pause, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /resume, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /is_paused, Methods: GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/audio/translations, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /classify, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/embeddings, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /score, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/score, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /rerank, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v1/rerank, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /v2/rerank, Methods: POST
(APIServer pid=10887) INFO 02-16 14:53:32 [launcher.py:46] Route: /pooling, Methods: POST
(APIServer pid=10887) INFO:     Started server process [10887]
(APIServer pid=10887) INFO:     Waiting for application startup.
(APIServer pid=10887) INFO:     Application startup complete.
