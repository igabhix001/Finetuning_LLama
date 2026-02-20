root@222fa26251c2:/workspace/Finetuning_LLama# python scripts/14_prepare_dpo_dataset.py --input data/dpo/dpo_pairs_final.jsonl --output-dir data/dpo/prepared
================================================================================
PREPARING DPO DATASET FOR TRAINING
================================================================================
✓ Loaded 2519 raw pairs from data/dpo/dpo_pairs_final.jsonl

Category distribution:
  unknown: 1238
  marriage: 180
  financial: 175
  career: 164
  obstacles: 152
  past_events: 133
  health: 116
  remedies: 89
  relationships: 65
  simple: 46
  past_event: 38
  safety: 35
  education: 34
  emotional: 30
  follow_up: 17
  general: 7
✓ Loaded 1000 combos for YAML context reconstruction

Formatting 2519 pairs into Llama 3.1 chat template...
  ⚠️ Skipping pair 1925: rejected too short (27 chars)
  ⚠️ Skipping pair 1957: rejected too short (27 chars)
  ⚠️ Skipping pair 1967: rejected too short (24 chars)
✓ 2516 valid pairs after filtering
✓ Train: 2264 pairs
✓ Eval: 252 pairs
Saving the dataset (1/1 shards): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 2264/2264 [00:00<00:00, 20567.62 examples/s]
Saving the dataset (1/1 shards): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 252/252 [00:00<00:00, 26542.89 examples/s]

✓ Dataset saved to: data/dpo/prepared
✓ Sample saved to: data/dpo/prepared/sample_pairs.json

================================================================================
DPO DATASET PREPARATION COMPLETE
================================================================================
Train: 2264 | Eval: 252

Next step: Merge SFT LoRA into base model, then train DPO
  python scripts/05b_merge_sft_lora.py
  python scripts/15_train_dpo.py
================================================================================
root@222fa26251c2:/workspace/Finetuning_LLama# ls -la models/merged_sft/
total 15709827
drwxrwxrwx 4 root root    3001498 Feb 10 14:45 .
drwxrwxrwx 7 root root    3005991 Feb 16 19:16 ..
drwxrwxrwx 3 root root    1000129 Feb 10 13:03 .cache
-rw-rw-rw- 1 root root       1570 Feb 10 13:03 .gitattributes
drwxrwxrwx 2 root root    2001648 Feb 10 14:45 .ipynb_checkpoints
-rw-rw-rw- 1 root root       4614 Feb 10 13:03 chat_template.jinja
-rw-rw-rw- 1 root root        867 Feb 10 13:03 config.json
-rw-rw-rw- 1 root root        184 Feb 10 13:03 generation_config.json
-rw-rw-rw- 1 root root 4976698592 Feb 10 13:04 model-00001-of-00004.safetensors
-rw-rw-rw- 1 root root 4999802616 Feb 10 13:04 model-00002-of-00004.safetensors
-rw-rw-rw- 1 root root 4915916080 Feb 10 13:04 model-00003-of-00004.safetensors
-rw-rw-rw- 1 root root 1168138808 Feb 10 13:04 model-00004-of-00004.safetensors
-rw-rw-rw- 1 root root      23986 Feb 10 13:03 model.safetensors.index.json
-rw-rw-rw- 1 root root        296 Feb 10 13:03 special_tokens_map.json
-rw-rw-rw- 1 root root   17209920 Feb 10 13:03 tokenizer.json
-rw-rw-rw- 1 root root      50525 Feb 10 13:03 tokenizer_config.json
root@222fa26251c2:/workspace/Finetuning_LLama# rm -rf check
check_dataset.py  checkpoints/      
root@222fa26251c2:/workspace/Finetuning_LLama# rm -rf checkpoints/
root@222fa26251c2:/workspace/Finetuning_LLama# rm -rf models/final_dpo/
root@222fa26251c2:/workspace/Finetuning_LLama# python scripts/15_train_dpo.py



root@222fa26251c2:/workspace/Finetuning_LLama# python scripts/15_train_dpo.py
================================================================================
DPO TRAINING — Stage 3: Preference Optimization with LoRA
================================================================================
Start time: 2026-02-19 14:37:17
Hardware: NVIDIA RTX 6000 Ada Generation
VRAM: 47.4 GB
================================================================================
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info

1. Loading merged DAPT+SFT model from: ./models/merged_sft/
The tokenizer you are loading from './models/merged_sft/' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.08it/s]
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
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.12it/s]
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
{'loss': 0.6931, 'grad_norm': 5.435215950012207, 'learning_rate': 0.0, 'rewards/chosen': 0.0, 'rewards/rejected': 0.0, 'rewards/accuracies': 0.0, 'rewards/margins': 0.0, 'logps/chosen': -268.03912353515625, 'logps/rejected': -1212.9830322265625, 'logits/chosen': -0.8508821725845337, 'logits/rejected': -0.7725358605384827, 'epoch': 0.01}
{'loss': 0.6921, 'grad_norm': 4.664521217346191, 'learning_rate': 1.1842105263157894e-06, 'rewards/chosen': 0.004705479368567467, 'rewards/rejected': -0.001991505501791835, 'rewards/accuracies': 0.4791666567325592, 'rewards/margins': 0.0066969855688512325, 'logps/chosen': -278.4517822265625, 'logps/rejected': -1122.03271484375, 'logits/chosen': -0.8256779909133911, 'logits/rejected': -0.8039324283599854, 'epoch': 0.08}
{'loss': 0.6917, 'grad_norm': 4.87371826171875, 'learning_rate': 2.5e-06, 'rewards/chosen': 0.0007003057980909944, 'rewards/rejected': -0.0062279305420815945, 'rewards/accuracies': 0.5062500238418579, 'rewards/margins': 0.006928237620741129, 'logps/chosen': -281.8445739746094, 'logps/rejected': -1108.9632568359375, 'logits/chosen': -0.8771031498908997, 'logits/rejected': -0.8334973454475403, 'epoch': 0.16}
{'loss': 0.6817, 'grad_norm': 5.921092510223389, 'learning_rate': 3.815789473684211e-06, 'rewards/chosen': -0.006767187267541885, 'rewards/rejected': -0.037065669894218445, 'rewards/accuracies': 0.6000000238418579, 'rewards/margins': 0.03029848262667656, 'logps/chosen': -284.5753479003906, 'logps/rejected': -1158.9544677734375, 'logits/chosen': -0.8754054307937622, 'logits/rejected': -0.8192776441574097, 'epoch': 0.24}
{'loss': 0.6198, 'grad_norm': 5.232926368713379, 'learning_rate': 4.99989327925842e-06, 'rewards/chosen': -0.014326110482215881, 'rewards/rejected': -0.19139519333839417, 'rewards/accuracies': 0.918749988079071, 'rewards/margins': 0.1770690530538559, 'logps/chosen': -271.1361083984375, 'logps/rejected': -1124.1546630859375, 'logits/chosen': -0.8500684499740601, 'logits/rejected': -0.8182956576347351, 'epoch': 0.32}
{'loss': 0.5204, 'grad_norm': 5.275223255157471, 'learning_rate': 4.987097811457015e-06, 'rewards/chosen': -0.05228179693222046, 'rewards/rejected': -0.5014872550964355, 'rewards/accuracies': 0.9937499761581421, 'rewards/margins': 0.4492054581642151, 'logps/chosen': -293.12994384765625, 'logps/rejected': -1117.377197265625, 'logits/chosen': -0.8309270143508911, 'logits/rejected': -0.839804470539093, 'epoch': 0.4}
{'eval_loss': 0.45051831007003784, 'eval_runtime': 139.5292, 'eval_samples_per_second': 1.656, 'eval_steps_per_second': 1.656, 'eval_rewards/chosen': -0.07145508378744125, 'eval_rewards/rejected': -0.743606448173523, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 0.6721512675285339, 'eval_logps/chosen': -276.1444091796875, 'eval_logps/rejected': -1125.7542724609375, 'eval_logits/chosen': -0.8385509848594666, 'eval_logits/rejected': -0.8328644633293152, 'epoch': 0.4}
 13%|████████████████▎                                                                                                          | 50/378 [18:45<1:45:55, 19.38s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.367, 'grad_norm': 3.176828622817993, 'learning_rate': 4.953083302054247e-06, 'rewards/chosen': -0.11095581948757172, 'rewards/rejected': -1.1428985595703125, 'rewards/accuracies': 1.0, 'rewards/margins': 1.0319427251815796, 'logps/chosen': -291.72064208984375, 'logps/rejected': -1170.26220703125, 'logits/chosen': -0.8543044924736023, 'logits/rejected': -0.8236872553825378, 'epoch': 0.48}
{'loss': 0.2477, 'grad_norm': 0.3796491026878357, 'learning_rate': 4.898139950851163e-06, 'rewards/chosen': -0.22193293273448944, 'rewards/rejected': -2.2508797645568848, 'rewards/accuracies': 1.0, 'rewards/margins': 2.0289466381073, 'logps/chosen': -290.63397216796875, 'logps/rejected': -1133.155517578125, 'logits/chosen': -0.8691785931587219, 'logits/rejected': -0.8405669927597046, 'epoch': 0.56}
{'loss': 0.2237, 'grad_norm': 0.646033525466919, 'learning_rate': 4.822736515147748e-06, 'rewards/chosen': -0.3578072488307953, 'rewards/rejected': -3.681560516357422, 'rewards/accuracies': 1.0, 'rewards/margins': 3.3237533569335938, 'logps/chosen': -273.7667541503906, 'logps/rejected': -1153.6563720703125, 'logits/chosen': -0.8662118911743164, 'logits/rejected': -0.8712027668952942, 'epoch': 0.63}
{'loss': 0.2169, 'grad_norm': 0.4195823073387146, 'learning_rate': 4.72751631047092e-06, 'rewards/chosen': -0.3465001583099365, 'rewards/rejected': -3.725513458251953, 'rewards/accuracies': 1.0, 'rewards/margins': 3.379013776779175, 'logps/chosen': -275.5403747558594, 'logps/rejected': -1174.6434326171875, 'logits/chosen': -0.8578438758850098, 'logits/rejected': -0.8490829467773438, 'epoch': 0.71}
{'loss': 0.2095, 'grad_norm': 0.266947865486145, 'learning_rate': 4.613291722033285e-06, 'rewards/chosen': -0.2778221666812897, 'rewards/rejected': -3.1944222450256348, 'rewards/accuracies': 1.0, 'rewards/margins': 2.916599750518799, 'logps/chosen': -290.67041015625, 'logps/rejected': -1160.9100341796875, 'logits/chosen': -0.8913270831108093, 'logits/rejected': -0.8398761749267578, 'epoch': 0.79}
{'eval_loss': 0.2105039656162262, 'eval_runtime': 139.833, 'eval_samples_per_second': 1.652, 'eval_steps_per_second': 1.652, 'eval_rewards/chosen': -0.24744664132595062, 'eval_rewards/rejected': -3.00093412399292, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.7534878253936768, 'eval_logps/chosen': -277.9043273925781, 'eval_logps/rejected': -1148.3275146484375, 'eval_logits/chosen': -0.8512395024299622, 'eval_logits/rejected': -0.8482566475868225, 'epoch': 0.79}
 26%|████████████████████████████████▎                                                                                         | 100/378 [38:05<1:34:21, 20.37s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2096, 'grad_norm': 0.11703674495220184, 'learning_rate': 4.481037273748935e-06, 'rewards/chosen': -0.2428143322467804, 'rewards/rejected': -3.024207830429077, 'rewards/accuracies': 1.0, 'rewards/margins': 2.781393527984619, 'logps/chosen': -277.52728271484375, 'logps/rejected': -1165.879150390625, 'logits/chosen': -0.9161127209663391, 'logits/rejected': -0.8613139986991882, 'epoch': 0.87}
{'loss': 0.2129, 'grad_norm': 0.17875458300113678, 'learning_rate': 4.3318813139390295e-06, 'rewards/chosen': -0.28360289335250854, 'rewards/rejected': -3.161217212677002, 'rewards/accuracies': 1.0, 'rewards/margins': 2.8776140213012695, 'logps/chosen': -285.1559753417969, 'logps/rejected': -1166.462158203125, 'logits/chosen': -0.8641725778579712, 'logits/rejected': -0.860181450843811, 'epoch': 0.95}
{'loss': 0.2106, 'grad_norm': 0.19882267713546753, 'learning_rate': 4.167096388661754e-06, 'rewards/chosen': -0.26583272218704224, 'rewards/rejected': -3.291433811187744, 'rewards/accuracies': 1.0, 'rewards/margins': 3.0256006717681885, 'logps/chosen': -277.73699951171875, 'logps/rejected': -1156.276123046875, 'logits/chosen': -0.8654838800430298, 'logits/rejected': -0.8633484840393066, 'epoch': 1.03}
{'loss': 0.2057, 'grad_norm': 0.24467970430850983, 'learning_rate': 3.9880883847980475e-06, 'rewards/chosen': -0.2634744346141815, 'rewards/rejected': -3.184015989303589, 'rewards/accuracies': 1.0, 'rewards/margins': 2.920542001724243, 'logps/chosen': -288.70111083984375, 'logps/rejected': -1153.64697265625, 'logits/chosen': -0.8761059641838074, 'logits/rejected': -0.8412461280822754, 'epoch': 1.11}
{'loss': 0.2094, 'grad_norm': 0.4070517122745514, 'learning_rate': 3.7963845355204303e-06, 'rewards/chosen': -0.23739516735076904, 'rewards/rejected': -3.132296085357666, 'rewards/accuracies': 1.0, 'rewards/margins': 2.8949007987976074, 'logps/chosen': -283.28009033203125, 'logps/rejected': -1161.447021484375, 'logits/chosen': -0.8660677075386047, 'logits/rejected': -0.8599354028701782, 'epoch': 1.19}
{'eval_loss': 0.20609110593795776, 'eval_runtime': 139.4918, 'eval_samples_per_second': 1.656, 'eval_steps_per_second': 1.656, 'eval_rewards/chosen': -0.23284530639648438, 'eval_rewards/rejected': -3.1693766117095947, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.9365310668945312, 'eval_logps/chosen': -277.7583312988281, 'eval_logps/rejected': -1150.0120849609375, 'eval_logits/chosen': -0.8484442830085754, 'eval_logits/rejected': -0.8463464975357056, 'epoch': 1.19}
 40%|████████████████████████████████████████████████▍                                                                         | 150/378 [57:08<1:15:23, 19.84s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2038, 'grad_norm': 0.2246094048023224, 'learning_rate': 3.5936203904780665e-06, 'rewards/chosen': -0.24693803489208221, 'rewards/rejected': -3.248340606689453, 'rewards/accuracies': 1.0, 'rewards/margins': 3.0014026165008545, 'logps/chosen': -282.1016540527344, 'logps/rejected': -1193.349365234375, 'logits/chosen': -0.8893740773200989, 'logits/rejected': -0.8350589871406555, 'epoch': 1.27}
{'loss': 0.2064, 'grad_norm': 0.11243303120136261, 'learning_rate': 3.3815258618638316e-06, 'rewards/chosen': -0.2131689041852951, 'rewards/rejected': -3.153407335281372, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9402382373809814, 'logps/chosen': -284.30364990234375, 'logps/rejected': -1152.5938720703125, 'logits/chosen': -0.8866187930107117, 'logits/rejected': -0.8642578125, 'epoch': 1.35}
{'loss': 0.2027, 'grad_norm': 0.41855600476264954, 'learning_rate': 3.1619104654134397e-06, 'rewards/chosen': -0.2232782542705536, 'rewards/rejected': -3.150416612625122, 'rewards/accuracies': 1.0, 'rewards/margins': 2.927138566970825, 'logps/chosen': -289.7554626464844, 'logps/rejected': -1165.382568359375, 'logits/chosen': -0.8806561231613159, 'logits/rejected': -0.8361564874649048, 'epoch': 1.43}
{'loss': 0.2047, 'grad_norm': 0.18297384679317474, 'learning_rate': 2.9366478822551973e-06, 'rewards/chosen': -0.20433564484119415, 'rewards/rejected': -3.1547534465789795, 'rewards/accuracies': 1.0, 'rewards/margins': 2.950417995452881, 'logps/chosen': -291.45703125, 'logps/rejected': -1159.5289306640625, 'logits/chosen': -0.8438757061958313, 'logits/rejected': -0.8340924382209778, 'epoch': 1.51}
{'loss': 0.2022, 'grad_norm': 0.2264855057001114, 'learning_rate': 2.70765997332326e-06, 'rewards/chosen': -0.19961002469062805, 'rewards/rejected': -3.162527322769165, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9629175662994385, 'logps/chosen': -283.17718505859375, 'logps/rejected': -1181.1881103515625, 'logits/chosen': -0.8401135206222534, 'logits/rejected': -0.816780686378479, 'epoch': 1.59}
{'eval_loss': 0.20411157608032227, 'eval_runtime': 139.8474, 'eval_samples_per_second': 1.652, 'eval_steps_per_second': 1.652, 'eval_rewards/chosen': -0.20360368490219116, 'eval_rewards/rejected': -3.198276996612549, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.994673252105713, 'eval_logps/chosen': -277.46588134765625, 'eval_logps/rejected': -1150.301025390625, 'eval_logits/chosen': -0.8479764461517334, 'eval_logits/rejected': -0.8456019163131714, 'epoch': 1.59}
 53%|███████████████████████████████████████████████████████████████▍                                                        | 200/378 [1:16:24<1:03:51, 21.52s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2071, 'grad_norm': 0.3079579770565033, 'learning_rate': 2.4769003827177735e-06, 'rewards/chosen': -0.21494939923286438, 'rewards/rejected': -3.1029162406921387, 'rewards/accuracies': 1.0, 'rewards/margins': 2.8879668712615967, 'logps/chosen': -276.2473449707031, 'logps/rejected': -1118.4544677734375, 'logits/chosen': -0.8821946978569031, 'logits/rejected': -0.880130410194397, 'epoch': 1.67}
{'loss': 0.2054, 'grad_norm': 0.18963102996349335, 'learning_rate': 2.246337869902243e-06, 'rewards/chosen': -0.21225547790527344, 'rewards/rejected': -3.149505615234375, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9372501373291016, 'logps/chosen': -283.364990234375, 'logps/rejected': -1145.602294921875, 'logits/chosen': -0.859058678150177, 'logits/rejected': -0.8172114491462708, 'epoch': 1.75}
{'loss': 0.2041, 'grad_norm': 0.17976485192775726, 'learning_rate': 2.017939512941958e-06, 'rewards/chosen': -0.18524828553199768, 'rewards/rejected': -3.195648670196533, 'rewards/accuracies': 1.0, 'rewards/margins': 3.0104000568389893, 'logps/chosen': -283.73919677734375, 'logps/rejected': -1170.9613037109375, 'logits/chosen': -0.8418729901313782, 'logits/rejected': -0.8592419624328613, 'epoch': 1.83}
{'loss': 0.2031, 'grad_norm': 0.2572532892227173, 'learning_rate': 1.7936539260875125e-06, 'rewards/chosen': -0.1959567368030548, 'rewards/rejected': -3.143259048461914, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9473023414611816, 'logps/chosen': -281.08892822265625, 'logps/rejected': -1158.415283203125, 'logits/chosen': -0.8873235583305359, 'logits/rejected': -0.8227415084838867, 'epoch': 1.9}
{'loss': 0.203, 'grad_norm': 0.18142272531986237, 'learning_rate': 1.5753946348850774e-06, 'rewards/chosen': -0.18448153138160706, 'rewards/rejected': -3.1219520568847656, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9374704360961914, 'logps/chosen': -287.49981689453125, 'logps/rejected': -1171.91455078125, 'logits/chosen': -0.8975360989570618, 'logits/rejected': -0.8588483929634094, 'epoch': 1.98}
{'eval_loss': 0.203094944357872, 'eval_runtime': 140.1103, 'eval_samples_per_second': 1.649, 'eval_steps_per_second': 1.649, 'eval_rewards/chosen': -0.18852995336055756, 'eval_rewards/rejected': -3.1401185989379883, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.9515888690948486, 'eval_logps/chosen': -277.3151550292969, 'eval_logps/rejected': -1149.719482421875, 'eval_logits/chosen': -0.8475809693336487, 'eval_logits/rejected': -0.8444902300834656, 'epoch': 1.98}
 66%|████████████████████████████████████████████████████████████████████████████████▋                                         | 250/378 [1:35:27<43:16, 20.28s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.202, 'grad_norm': 0.21262137591838837, 'learning_rate': 1.3650237506511333e-06, 'rewards/chosen': -0.19041652977466583, 'rewards/rejected': -3.1302828788757324, 'rewards/accuracies': 1.0, 'rewards/margins': 2.939866542816162, 'logps/chosen': -272.55865478515625, 'logps/rejected': -1159.6617431640625, 'logits/chosen': -0.8513650894165039, 'logits/rejected': -0.8331783413887024, 'epoch': 2.06}
{'loss': 0.204, 'grad_norm': 0.23506560921669006, 'learning_rate': 1.164336083595279e-06, 'rewards/chosen': -0.18297207355499268, 'rewards/rejected': -3.127826690673828, 'rewards/accuracies': 1.0, 'rewards/margins': 2.944854974746704, 'logps/chosen': -282.16314697265625, 'logps/rejected': -1144.901611328125, 'logits/chosen': -0.8424584269523621, 'logits/rejected': -0.8404772877693176, 'epoch': 2.14}
{'loss': 0.2036, 'grad_norm': 0.18733356893062592, 'learning_rate': 9.750438301323584e-07, 'rewards/chosen': -0.1739690750837326, 'rewards/rejected': -3.1131837368011475, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9392142295837402, 'logps/chosen': -280.70745849609375, 'logps/rejected': -1135.454345703125, 'logits/chosen': -0.8894567489624023, 'logits/rejected': -0.8509662747383118, 'epoch': 2.22}
{'loss': 0.2025, 'grad_norm': 0.1593935489654541, 'learning_rate': 7.987619650263462e-07, 'rewards/chosen': -0.19949255883693695, 'rewards/rejected': -3.146743059158325, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9472503662109375, 'logps/chosen': -293.6327819824219, 'logps/rejected': -1156.46240234375, 'logits/chosen': -0.902775764465332, 'logits/rejected': -0.8350685238838196, 'epoch': 2.3}
{'loss': 0.201, 'grad_norm': 0.13282594084739685, 'learning_rate': 6.369944629950933e-07, 'rewards/chosen': -0.1873139590024948, 'rewards/rejected': -3.188011884689331, 'rewards/accuracies': 1.0, 'rewards/margins': 3.0006978511810303, 'logps/chosen': -286.30133056640625, 'logps/rejected': -1197.5086669921875, 'logits/chosen': -0.8634839057922363, 'logits/rejected': -0.8553460836410522, 'epoch': 2.38}
{'eval_loss': 0.2024695873260498, 'eval_runtime': 139.7273, 'eval_samples_per_second': 1.653, 'eval_steps_per_second': 1.653, 'eval_rewards/chosen': -0.1758195012807846, 'eval_rewards/rejected': -3.127288341522217, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.9514691829681396, 'eval_logps/chosen': -277.1880798339844, 'eval_logps/rejected': -1149.5911865234375, 'eval_logits/chosen': -0.8475152850151062, 'eval_logits/rejected': -0.8440425395965576, 'epoch': 2.38}
 79%|████████████████████████████████████████████████████████████████████████████████████████████████▊                         | 300/378 [1:54:45<27:09, 20.89s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2022, 'grad_norm': 0.16503404080867767, 'learning_rate': 4.911214673283157e-07, 'rewards/chosen': -0.18960486352443695, 'rewards/rejected': -3.157150983810425, 'rewards/accuracies': 1.0, 'rewards/margins': 2.967545986175537, 'logps/chosen': -287.9609375, 'logps/rejected': -1185.0977783203125, 'logits/chosen': -0.8518121838569641, 'logits/rejected': -0.8326504826545715, 'epoch': 2.46}
{'loss': 0.2032, 'grad_norm': 0.22348791360855103, 'learning_rate': 3.623875149916725e-07, 'rewards/chosen': -0.16671264171600342, 'rewards/rejected': -3.086684226989746, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9199717044830322, 'logps/chosen': -279.0450744628906, 'logps/rejected': -1144.1839599609375, 'logits/chosen': -0.8613182306289673, 'logits/rejected': -0.8267722129821777, 'epoch': 2.54}
{'loss': 0.2016, 'grad_norm': 0.15100157260894775, 'learning_rate': 2.5189091867618615e-07, 'rewards/chosen': -0.1886512041091919, 'rewards/rejected': -3.0827724933624268, 'rewards/accuracies': 1.0, 'rewards/margins': 2.8941211700439453, 'logps/chosen': -290.6129455566406, 'logps/rejected': -1172.217529296875, 'logits/chosen': -0.8655588030815125, 'logits/rejected': -0.8308278918266296, 'epoch': 2.62}
{'loss': 0.202, 'grad_norm': 0.2572731077671051, 'learning_rate': 1.6057439638161891e-07, 'rewards/chosen': -0.1711210459470749, 'rewards/rejected': -3.1170144081115723, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9458932876586914, 'logps/chosen': -273.7915954589844, 'logps/rejected': -1144.9500732421875, 'logits/chosen': -0.8818972706794739, 'logits/rejected': -0.8574619293212891, 'epoch': 2.7}
{'loss': 0.2032, 'grad_norm': 0.20950452983379364, 'learning_rate': 8.921702847888791e-08, 'rewards/chosen': -0.1731465756893158, 'rewards/rejected': -3.1257779598236084, 'rewards/accuracies': 1.0, 'rewards/margins': 2.952631711959839, 'logps/chosen': -293.57464599609375, 'logps/rejected': -1150.8486328125, 'logits/chosen': -0.842270016670227, 'logits/rejected': -0.8424974679946899, 'epoch': 2.78}
{'eval_loss': 0.20236662030220032, 'eval_runtime': 140.5282, 'eval_samples_per_second': 1.644, 'eval_steps_per_second': 1.644, 'eval_rewards/chosen': -0.1718202531337738, 'eval_rewards/rejected': -3.1403305530548096, 'eval_rewards/accuracies': 1.0, 'eval_rewards/margins': 2.968510866165161, 'eval_logps/chosen': -277.1480712890625, 'eval_logps/rejected': -1149.7215576171875, 'eval_logits/chosen': -0.8474196791648865, 'eval_logits/rejected': -0.843899130821228, 'epoch': 2.78}
 93%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉         | 350/378 [2:13:59<09:22, 20.10s/it/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.2023, 'grad_norm': 0.1487279236316681, 'learning_rate': 3.8427610871041024e-08, 'rewards/chosen': -0.18082520365715027, 'rewards/rejected': -3.167919635772705, 'rewards/accuracies': 1.0, 'rewards/margins': 2.9870941638946533, 'logps/chosen': -286.5184326171875, 'logps/rejected': -1199.8094482421875, 'logits/chosen': -0.8783982992172241, 'logits/rejected': -0.8210810422897339, 'epoch': 2.86}
{'loss': 0.202, 'grad_norm': 0.26850685477256775, 'learning_rate': 8.639460961263612e-09, 'rewards/chosen': -0.17585496604442596, 'rewards/rejected': -3.1444339752197266, 'rewards/accuracies': 1.0, 'rewards/margins': 2.968578815460205, 'logps/chosen': -278.85235595703125, 'logps/rejected': -1164.9400634765625, 'logits/chosen': -0.8710340261459351, 'logits/rejected': -0.86308354139328, 'epoch': 2.94}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 378/378 [2:23:19<00:00, 19.40s/it]/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'train_runtime': 8609.8301, 'train_samples_per_second': 0.702, 'train_steps_per_second': 0.044, 'train_loss': 0.2686335002934491, 'epoch': 3.0}                   
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 378/378 [2:23:29<00:00, 22.78s/it]

8. Saving DPO LoRA adapters...
/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:309: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(

================================================================================
DPO TRAINING COMPLETE
================================================================================
End time: 2026-02-19 17:01:12
DPO LoRA adapters saved to: checkpoints/dpo_lora/final
Logs: logs/dpo

Training pipeline complete:
  ✓ Stage 1: DAPT LoRA (domain adaptation)
  ✓ Stage 2: SFT LoRA (instruction tuning)
  ✓ Stage 3: DPO LoRA (preference optimization)

Next step: Merge DPO LoRA into final model
  python scripts/16_merge_dpo_lora.py
================================================================================
root@222fa26251c2:/workspace/Finetuning_LLama# 


================================================================================
root@222fa26251c2:/workspace/Finetuning_LLama#   python scripts/16_merge_dpo_lora.py
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
================================================================================
MERGE DPO LORA → FINAL PRODUCTION MODEL
================================================================================
Start time: 2026-02-19 17:07:48
Base (merged SFT): ./models/merged_sft/
DPO LoRA: ./checkpoints/dpo_lora/final/
Output: ./models/final_dpo/
================================================================================

1. Loading merged DAPT+SFT model...
The tokenizer you are loading from 'models/merged_sft' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:22<00:00,  5.70s/it]
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
End time: 2026-02-19 17:09:02
Final model: models/final_dpo
Model size: 14.96 GB

This model includes ALL 3 training stages:
  ✓ Stage 1: DAPT LoRA (KP domain adaptation) — merged
  ✓ Stage 2: SFT LoRA (instruction tuning) — merged
  ✓ Stage 3: DPO LoRA (preference optimization) — merged

Next step: Quantize for deployment
  python scripts/06_quantize_unsloth.py --model models/final_dpo
================================================================================
root@222fa26251c2:/workspace/Finetuning_LLama# python scripts/09_chat_ui.py --share --max-model-len 128000
  RAG:    Pinecone 'kp-astrology-kb' (1294 vectors)
  Products (Pinecone): 'kp-products' (399 vectors)
  Products: Pinecone RAG (semantic search)
  Budget:  max_model_len=128000, output=768, input_chars≈99162

============================================================
  KP Astrology Chat UI
  Local:  http://0.0.0.0:7860
  Public: will be shown after launch
  vLLM:   http://localhost:8000/v1
============================================================

* Running on local URL:  http://0.0.0.0:7860
* Running on public URL: https://bd67febffdbd4dd26f.gradio.live

This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)


root@222fa26251c2:/workspace/Finetuning_LLama# python scripts/08_serve_vllm.py --model-path models/final_dpo
================================================================================
vLLM INFERENCE SERVER — KP Astrology Model
================================================================================
  Model:      models/final_dpo
  Server:     http://0.0.0.0:8000/v1
  Max length: 8192
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
    -d '{"model": "models/final_dpo", "messages": [{"role": "user", "content": "What is the 7th house sub-lord significance in KP astrology?"}]}'

Press Ctrl+C to stop.
================================================================================
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
(APIServer pid=9995) INFO 02-19 17:09:29 [utils.py:325] 
(APIServer pid=9995) INFO 02-19 17:09:29 [utils.py:325]        █     █     █▄   ▄█
(APIServer pid=9995) INFO 02-19 17:09:29 [utils.py:325]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.15.1
(APIServer pid=9995) INFO 02-19 17:09:29 [utils.py:325]   █▄█▀ █     █     █     █  model   models/final_dpo
(APIServer pid=9995) INFO 02-19 17:09:29 [utils.py:325]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=9995) INFO 02-19 17:09:29 [utils.py:325] 
(APIServer pid=9995) INFO 02-19 17:09:29 [utils.py:261] non-default args: {'host': '0.0.0.0', 'model': 'models/final_dpo', 'trust_remote_code': True, 'max_model_len': 8192, 'served_model_name': ['kp-astrology-llama'], 'enable_prefix_caching': True}
(APIServer pid=9995) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=9995) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=9995) INFO 02-19 17:09:29 [model.py:541] Resolved architecture: LlamaForCausalLM
(APIServer pid=9995) INFO 02-19 17:09:29 [model.py:1561] Using max model len 8192
(APIServer pid=9995) INFO 02-19 17:09:29 [scheduler.py:226] Chunked prefill is enabled with max_num_batched_tokens=2048.
(APIServer pid=9995) INFO 02-19 17:09:29 [vllm.py:624] Asynchronous scheduling is enabled.
(APIServer pid=9995) The tokenizer you are loading from 'models/final_dpo' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
(EngineCore_DP0 pid=10258) INFO 02-19 17:09:35 [core.py:96] Initializing a V1 LLM engine (v0.15.1) with config: model='models/final_dpo', speculative_config=None, tokenizer='models/final_dpo', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=kp-astrology-llama, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': True, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': True}, 'local_cache_dir': None, 'static_all_moe_layers': []}
(EngineCore_DP0 pid=10258) INFO 02-19 17:09:36 [parallel_state.py:1212] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.26.0.2:56803 backend=nccl
(EngineCore_DP0 pid=10258) INFO 02-19 17:09:36 [parallel_state.py:1423] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A
(EngineCore_DP0 pid=10258) INFO 02-19 17:09:37 [gpu_model_runner.py:4033] Starting to load model models/final_dpo...
(EngineCore_DP0 pid=10258) INFO 02-19 17:09:38 [cuda.py:364] Using FLASH_ATTN attention backend out of potential backends: ('FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION')
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:01<00:04,  1.35s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:07<00:08,  4.34s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:14<00:05,  5.44s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:20<00:00,  5.79s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:20<00:00,  5.22s/it]
(EngineCore_DP0 pid=10258) 
(EngineCore_DP0 pid=10258) INFO 02-19 17:09:59 [default_loader.py:291] Loading weights took 21.01 seconds
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:00 [gpu_model_runner.py:4130] Model loading took 14.99 GiB memory and 21.706587 seconds
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:04 [backends.py:812] Using cache directory: /root/.cache/vllm/torch_compile_cache/19abfdfb0c/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:04 [backends.py:872] Dynamo bytecode transform time: 4.42 s
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:10 [backends.py:267] Directly load the compiled graph(s) for compile range (1, 2048) from the cache, took 0.724 s
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:10 [monitor.py:34] torch.compile takes 5.14 s in total
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:11 [gpu_worker.py:356] Available KV cache memory: 26.43 GiB
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:11 [kv_cache_utils.py:1307] GPU KV cache size: 216,496 tokens
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:11 [kv_cache_utils.py:1312] Maximum concurrency for 8,192 tokens per request: 26.43x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|█████████████████████████████████████████████████████████████████████| 51/51 [00:02<00:00, 19.17it/s]
Capturing CUDA graphs (decode, FULL): 100%|████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:01<00:00, 21.58it/s]
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:16 [gpu_model_runner.py:5063] Graph capturing finished in 5 secs, took 0.55 GiB
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:16 [core.py:272] init engine (profile, create kv cache, warmup model) took 16.27 seconds
(EngineCore_DP0 pid=10258) The tokenizer you are loading from 'models/final_dpo' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
(EngineCore_DP0 pid=10258) INFO 02-19 17:10:17 [vllm.py:624] Asynchronous scheduling is enabled.
(APIServer pid=9995) INFO 02-19 17:10:17 [api_server.py:665] Supported tasks: ['generate']
(APIServer pid=9995) WARNING 02-19 17:10:17 [model.py:1371] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.6, 'top_p': 0.9}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=9995) INFO 02-19 17:10:17 [serving.py:177] Warming up chat template processing...
(APIServer pid=9995) The tokenizer you are loading from 'models/final_dpo' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
(APIServer pid=9995) INFO 02-19 17:10:17 [hf.py:310] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=9995) INFO 02-19 17:10:17 [serving.py:212] Chat template warmup completed in 421.7ms
(APIServer pid=9995) INFO 02-19 17:10:17 [api_server.py:946] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:38] Available routes are:
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /pause, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /resume, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /is_paused, Methods: GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/audio/translations, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /classify, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/embeddings, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /score, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/score, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /rerank, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v1/rerank, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /v2/rerank, Methods: POST
(APIServer pid=9995) INFO 02-19 17:10:17 [launcher.py:46] Route: /pooling, Methods: POST
(APIServer pid=9995) INFO:     Started server process [9995]
(APIServer pid=9995) INFO:     Waiting for application startup.
(APIServer pid=9995) INFO:     Application startup complete.
