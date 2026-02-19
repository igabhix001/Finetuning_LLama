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
* Running on public URL: https://53e340c499ae29e1c4.gradio.live

This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
{"ts": "2026-02-19T10:18:20.584133Z", "event": "chat_response", "req_id": "ea67718d2c8c", "query_type": "simple", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 150, "temperature": 0.3, "raw_len": 617, "answer_len": 101, "latency_ms": 4266}
{"ts": "2026-02-19T10:18:38.434138Z", "event": "chat_response", "req_id": "7bd29ad0889d", "query_type": "simple", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 150, "temperature": 0.3, "raw_len": 667, "answer_len": 89, "latency_ms": 3921}
{"ts": "2026-02-19T10:18:54.026309Z", "event": "chat_response", "req_id": "c647efc8c715", "query_type": "simple", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 150, "temperature": 0.3, "raw_len": 679, "answer_len": 99, "latency_ms": 3236}
{"ts": "2026-02-19T10:19:23.937058Z", "event": "deflection_detected", "req_id": "feb302f8cee2", "original": "Arjun Mehta ji, aapki marriage timing ke liye planetary positions aur current dasha sequence dekhte hue, main indicate karunga ki marriage un planets ke combined influence mein manifest hogi jo houses"}
{"ts": "2026-02-19T10:19:32.165672Z", "event": "chat_response", "req_id": "feb302f8cee2", "query_type": "timing", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 450, "temperature": 0.5, "raw_len": 2170, "answer_len": 394, "latency_ms": 15595}
{"ts": "2026-02-19T10:21:11.579074Z", "event": "chat_response", "req_id": "c65fe5a1876c", "query_type": "timing", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 450, "temperature": 0.5, "raw_len": 1784, "answer_len": 288, "latency_ms": 8957}
{"ts": "2026-02-19T10:21:48.263901Z", "event": "chat_response", "req_id": "e1b8fcfa9626", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 1479, "answer_len": 595, "latency_ms": 7625}
{"ts": "2026-02-19T10:22:29.170590Z", "event": "chat_response", "req_id": "a02174ae1e9b", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 1390, "answer_len": 547, "latency_ms": 6853}
{"ts": "2026-02-19T10:23:05.902116Z", "event": "chat_response", "req_id": "82c86eeb63b9", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 4524, "answer_len": 492, "latency_ms": 14365}
{"ts": "2026-02-19T10:23:39.384469Z", "event": "chat_response", "req_id": "57204e6dec66", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 2913, "answer_len": 633, "latency_ms": 15570}
{"ts": "2026-02-19T10:25:15.750302Z", "event": "deflection_detected", "req_id": "0faeadc2f20c", "original": "New Career Opportunity Timing\n\nCurrently running Venus-Mercury pratyantar within Venus mahadasha, jo favorable conditions create karta hai employment changes ke liye. However, precise timing depend ka"}
{"ts": "2026-02-19T10:25:23.989712Z", "event": "chat_response", "req_id": "0faeadc2f20c", "query_type": "timing", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 450, "temperature": 0.5, "raw_len": 2228, "answer_len": 519, "latency_ms": 14408}
{"ts": "2026-02-19T10:25:37.971340Z", "event": "chat_response", "req_id": "38149347f0a4", "query_type": "timing", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 450, "temperature": 0.5, "raw_len": 1142, "answer_len": 370, "latency_ms": 4942}
{"ts": "2026-02-19T10:26:00.303308Z", "event": "chat_response", "req_id": "ed63e9184256", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 4005, "answer_len": 437, "latency_ms": 15147}
{"ts": "2026-02-19T10:26:09.197919Z", "event": "chat_response", "req_id": "b2cd92d2de4b", "query_type": "timing", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 450, "temperature": 0.5, "raw_len": 1409, "answer_len": 538, "latency_ms": 6127}
{"ts": "2026-02-19T10:26:27.616758Z", "event": "deflection_detected", "req_id": "0f38f883d06e", "original": "Life events in 2020 corresponded to Ketu-Mercury anthardasha period within Venus mahadasha. Since Ketu represents past karma and Mercury governs communication/intelligence, this combination brought op"}
{"ts": "2026-02-19T10:26:38.537263Z", "event": "chat_response", "req_id": "0f38f883d06e", "query_type": "past_event", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 600, "temperature": 0.4, "raw_len": 2826, "answer_len": 907, "latency_ms": 22302}
{"ts": "2026-02-19T10:27:30.846831Z", "event": "chat_response", "req_id": "a683b01029e2", "query_type": "past_event", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 600, "temperature": 0.4, "raw_len": 1309, "answer_len": 656, "latency_ms": 6478}
{"ts": "2026-02-19T10:27:39.587722Z", "event": "deflection_detected", "req_id": "34be17640621", "original": "First job aapne Ketu-Jupiter antardasha mein September 2006 se August 2007 ke beech secure kiya tha. Is period mein, Jupiter as 2nd, 6th, 10th cusp sub-lord strongly signify karta hai houses 2,6,10 jo"}
{"ts": "2026-02-19T10:27:50.489149Z", "event": "chat_response", "req_id": "34be17640621", "query_type": "past_event", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 600, "temperature": 0.4, "raw_len": 2250, "answer_len": 746, "latency_ms": 17319}
{"ts": "2026-02-19T10:28:07.886891Z", "event": "chat_response", "req_id": "1fb16658cac3", "query_type": "emotional", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 500, "temperature": 0.4, "raw_len": 1995, "answer_len": 584, "latency_ms": 9676}
{"ts": "2026-02-19T10:28:18.590978Z", "event": "chat_response", "req_id": "70155911661c", "query_type": "emotional", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 500, "temperature": 0.4, "raw_len": 1377, "answer_len": 763, "latency_ms": 5829}
{"ts": "2026-02-19T10:28:31.262489Z", "event": "chat_response", "req_id": "8f45154b7b04", "query_type": "emotional", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 500, "temperature": 0.4, "raw_len": 1557, "answer_len": 728, "latency_ms": 6509}
{"ts": "2026-02-19T10:28:48.818721Z", "event": "chat_response", "req_id": "a8915fdcfa7d", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 2271, "answer_len": 679, "latency_ms": 15059}
{"ts": "2026-02-19T10:29:01.653285Z", "event": "chat_response", "req_id": "8071c9a9e22f", "query_type": "emotional", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 500, "temperature": 0.4, "raw_len": 2066, "answer_len": 607, "latency_ms": 9631}
{"ts": "2026-02-19T10:29:29.331138Z", "event": "chat_response", "req_id": "d16315bc4b14", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 3473, "answer_len": 1664, "latency_ms": 14260}
{"ts": "2026-02-19T10:29:45.675990Z", "event": "chat_response", "req_id": "e722cb2850ea", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 3804, "answer_len": 617, "latency_ms": 14435}
{"ts": "2026-02-19T10:30:15.746207Z", "event": "chat_response", "req_id": "19b59b8d8f4b", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 2364, "answer_len": 613, "latency_ms": 14412}
{"ts": "2026-02-19T10:30:34.090761Z", "event": "chat_response", "req_id": "f8b1ab11a22c", "query_type": "remedy", "is_remedy": true, "has_chart": true, "rag_chunks": 5, "max_tokens": 500, "temperature": 0.5, "raw_len": 2255, "answer_len": 772, "latency_ms": 10409}
{"ts": "2026-02-19T10:30:51.960541Z", "event": "chat_response", "req_id": "44701cc1896d", "query_type": "analysis", "is_remedy": false, "has_chart": true, "rag_chunks": 5, "max_tokens": 768, "temperature": 0.5, "raw_len": 2587, "answer_len": 554, "latency_ms": 14410}




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
(APIServer pid=1726) INFO 02-19 10:10:57 [utils.py:325] 
(APIServer pid=1726) INFO 02-19 10:10:57 [utils.py:325]        █     █     █▄   ▄█
(APIServer pid=1726) INFO 02-19 10:10:57 [utils.py:325]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.15.1
(APIServer pid=1726) INFO 02-19 10:10:57 [utils.py:325]   █▄█▀ █     █     █     █  model   models/final_dpo
(APIServer pid=1726) INFO 02-19 10:10:57 [utils.py:325]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=1726) INFO 02-19 10:10:57 [utils.py:325] 
(APIServer pid=1726) INFO 02-19 10:10:57 [utils.py:261] non-default args: {'host': '0.0.0.0', 'model': 'models/final_dpo', 'trust_remote_code': True, 'max_model_len': 8192, 'served_model_name': ['kp-astrology-llama'], 'enable_prefix_caching': True}
(APIServer pid=1726) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=1726) The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
(APIServer pid=1726) INFO 02-19 10:11:05 [model.py:541] Resolved architecture: LlamaForCausalLM
(APIServer pid=1726) INFO 02-19 10:11:05 [model.py:1561] Using max model len 8192
(APIServer pid=1726) INFO 02-19 10:11:06 [scheduler.py:226] Chunked prefill is enabled with max_num_batched_tokens=2048.
(APIServer pid=1726) INFO 02-19 10:11:06 [vllm.py:624] Asynchronous scheduling is enabled.
(APIServer pid=1726) The tokenizer you are loading from 'models/final_dpo' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
(EngineCore_DP0 pid=2120) INFO 02-19 10:11:12 [core.py:96] Initializing a V1 LLM engine (v0.15.1) with config: model='models/final_dpo', speculative_config=None, tokenizer='models/final_dpo', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=kp-astrology-llama, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'level': None, 'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update'], 'compile_mm_encoder': False, 'compile_sizes': [], 'compile_ranges_split_points': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'eliminate_noops': True, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': True}, 'local_cache_dir': None, 'static_all_moe_layers': []}
(EngineCore_DP0 pid=2120) INFO 02-19 10:11:14 [parallel_state.py:1212] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.26.0.2:38647 backend=nccl
(EngineCore_DP0 pid=2120) INFO 02-19 10:11:14 [parallel_state.py:1423] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A
(EngineCore_DP0 pid=2120) INFO 02-19 10:11:14 [gpu_model_runner.py:4033] Starting to load model models/final_dpo...
(EngineCore_DP0 pid=2120) INFO 02-19 10:11:32 [cuda.py:364] Using FLASH_ATTN attention backend out of potential backends: ('FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION')
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:01<00:04,  1.65s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:09<00:10,  5.09s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:16<00:06,  6.35s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:24<00:00,  6.72s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:24<00:00,  6.07s/it]
(EngineCore_DP0 pid=2120) 
(EngineCore_DP0 pid=2120) INFO 02-19 10:11:56 [default_loader.py:291] Loading weights took 24.45 seconds
(EngineCore_DP0 pid=2120) INFO 02-19 10:11:57 [gpu_model_runner.py:4130] Model loading took 14.99 GiB memory and 42.281021 seconds
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:02 [backends.py:812] Using cache directory: /root/.cache/vllm/torch_compile_cache/19abfdfb0c/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:02 [backends.py:872] Dynamo bytecode transform time: 4.21 s
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:09 [backends.py:302] Cache the graph of compile range (1, 2048) for later use
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:13 [backends.py:319] Compiling a graph for compile range (1, 2048) takes 7.29 s
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:13 [monitor.py:34] torch.compile takes 11.50 s in total
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:14 [gpu_worker.py:356] Available KV cache memory: 26.43 GiB
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:14 [kv_cache_utils.py:1307] GPU KV cache size: 216,512 tokens
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:14 [kv_cache_utils.py:1312] Maximum concurrency for 8,192 tokens per request: 26.43x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|█████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:02<00:00, 19.20it/s]
Capturing CUDA graphs (decode, FULL): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:01<00:00, 21.71it/s]
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:19 [gpu_model_runner.py:5063] Graph capturing finished in 5 secs, took 0.55 GiB
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:19 [core.py:272] init engine (profile, create kv cache, warmup model) took 22.34 seconds
(EngineCore_DP0 pid=2120) The tokenizer you are loading from 'models/final_dpo' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
(EngineCore_DP0 pid=2120) INFO 02-19 10:12:20 [vllm.py:624] Asynchronous scheduling is enabled.
(APIServer pid=1726) INFO 02-19 10:12:21 [api_server.py:665] Supported tasks: ['generate']
(APIServer pid=1726) WARNING 02-19 10:12:21 [model.py:1371] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.6, 'top_p': 0.9}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=1726) INFO 02-19 10:12:21 [serving.py:177] Warming up chat template processing...
(APIServer pid=1726) The tokenizer you are loading from 'models/final_dpo' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
(APIServer pid=1726) INFO 02-19 10:12:21 [hf.py:310] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=1726) INFO 02-19 10:12:21 [serving.py:212] Chat template warmup completed in 448.7ms
(APIServer pid=1726) INFO 02-19 10:12:21 [api_server.py:946] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:38] Available routes are:
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /pause, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /resume, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /is_paused, Methods: GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/audio/translations, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /classify, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/embeddings, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /score, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/score, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /rerank, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v1/rerank, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /v2/rerank, Methods: POST
(APIServer pid=1726) INFO 02-19 10:12:21 [launcher.py:46] Route: /pooling, Methods: POST
(APIServer pid=1726) INFO:     Started server process [1726]
(APIServer pid=1726) INFO:     Waiting for application startup.
(APIServer pid=1726) INFO:     Application startup complete.
(APIServer pid=1726) INFO:     127.0.0.1:49622 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:18:21 [loggers.py:257] Engine 000: Avg prompt throughput: 302.3 tokens/s, Avg generation throughput: 15.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1726) INFO 02-19 10:18:31 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1726) INFO:     127.0.0.1:49662 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:18:41 [loggers.py:257] Engine 000: Avg prompt throughput: 314.2 tokens/s, Avg generation throughput: 15.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 13.0%
(APIServer pid=1726) INFO:     127.0.0.1:52936 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:18:51 [loggers.py:257] Engine 000: Avg prompt throughput: 317.1 tokens/s, Avg generation throughput: 2.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.5%, Prefix cache hit rate: 17.1%
(APIServer pid=1726) INFO 02-19 10:19:01 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 12.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 17.1%
(APIServer pid=1726) INFO 02-19 10:19:11 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 17.1%
(APIServer pid=1726) INFO:     127.0.0.1:57334 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:19:21 [loggers.py:257] Engine 000: Avg prompt throughput: 324.6 tokens/s, Avg generation throughput: 25.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.6%, Prefix cache hit rate: 19.1%
(APIServer pid=1726) INFO 02-19 10:19:31 [loggers.py:257] Engine 000: Avg prompt throughput: 325.6 tokens/s, Avg generation throughput: 55.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.7%, Prefix cache hit rate: 22.9%
(APIServer pid=1726) INFO:     127.0.0.1:48950 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:19:41 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 22.9%
(APIServer pid=1726) INFO 02-19 10:19:51 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 22.9%
(APIServer pid=1726) INFO:     127.0.0.1:49390 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:21:11 [loggers.py:257] Engine 000: Avg prompt throughput: 333.2 tokens/s, Avg generation throughput: 45.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.1%
(APIServer pid=1726) INFO 02-19 10:21:21 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.1%
(APIServer pid=1726) INFO:     127.0.0.1:59958 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:21:51 [loggers.py:257] Engine 000: Avg prompt throughput: 328.7 tokens/s, Avg generation throughput: 33.8 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.3%
(APIServer pid=1726) INFO 02-19 10:22:01 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.3%
(APIServer pid=1726) INFO:     127.0.0.1:52336 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:22:31 [loggers.py:257] Engine 000: Avg prompt throughput: 347.7 tokens/s, Avg generation throughput: 29.9 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.3%
(APIServer pid=1726) INFO 02-19 10:22:41 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.3%
(APIServer pid=1726) INFO:     127.0.0.1:53040 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:23:01 [loggers.py:257] Engine 000: Avg prompt throughput: 359.4 tokens/s, Avg generation throughput: 54.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 23.5%
(APIServer pid=1726) INFO 02-19 10:23:11 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 22.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.5%
(APIServer pid=1726) INFO 02-19 10:23:21 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.5%
(APIServer pid=1726) INFO:     127.0.0.1:52162 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:23:31 [loggers.py:257] Engine 000: Avg prompt throughput: 353.6 tokens/s, Avg generation throughput: 34.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 23.4%
(APIServer pid=1726) INFO 02-19 10:23:41 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 42.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.4%
(APIServer pid=1726) INFO 02-19 10:23:51 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 23.4%
(APIServer pid=1726) INFO:     127.0.0.1:35414 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:25:11 [loggers.py:257] Engine 000: Avg prompt throughput: 349.9 tokens/s, Avg generation throughput: 7.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.7%, Prefix cache hit rate: 23.4%
(APIServer pid=1726) INFO 02-19 10:25:21 [loggers.py:257] Engine 000: Avg prompt throughput: 312.6 tokens/s, Avg generation throughput: 55.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.6%, Prefix cache hit rate: 24.3%
(APIServer pid=1726) INFO:     127.0.0.1:35424 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:25:31 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 11.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.3%
(APIServer pid=1726) INFO:     127.0.0.1:52368 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:25:41 [loggers.py:257] Engine 000: Avg prompt throughput: 334.6 tokens/s, Avg generation throughput: 24.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.3%
(APIServer pid=1726) INFO:     127.0.0.1:45708 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:25:51 [loggers.py:257] Engine 000: Avg prompt throughput: 352.4 tokens/s, Avg generation throughput: 29.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 24.2%
(APIServer pid=1726) INFO 02-19 10:26:01 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 47.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.2%
(APIServer pid=1726) INFO:     127.0.0.1:45080 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:26:11 [loggers.py:257] Engine 000: Avg prompt throughput: 335.4 tokens/s, Avg generation throughput: 30.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.1%
(APIServer pid=1726) INFO:     127.0.0.1:44078 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:26:21 [loggers.py:257] Engine 000: Avg prompt throughput: 358.4 tokens/s, Avg generation throughput: 27.8 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 24.2%
(APIServer pid=1726) INFO 02-19 10:26:31 [loggers.py:257] Engine 000: Avg prompt throughput: 327.1 tokens/s, Avg generation throughput: 54.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.6%, Prefix cache hit rate: 25.0%
(APIServer pid=1726) INFO:     127.0.0.1:60666 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:26:41 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 37.3 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 25.0%
(APIServer pid=1726) INFO 02-19 10:26:51 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 25.0%
(APIServer pid=1726) INFO:     127.0.0.1:33246 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:27:31 [loggers.py:257] Engine 000: Avg prompt throughput: 341.5 tokens/s, Avg generation throughput: 30.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.9%
(APIServer pid=1726) INFO:     127.0.0.1:49048 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:27:41 [loggers.py:257] Engine 000: Avg prompt throughput: 669.6 tokens/s, Avg generation throughput: 40.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.5%, Prefix cache hit rate: 25.2%
(APIServer pid=1726) INFO:     127.0.0.1:49064 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:27:51 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 48.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 25.2%
(APIServer pid=1726) INFO:     127.0.0.1:47476 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:28:01 [loggers.py:257] Engine 000: Avg prompt throughput: 370.4 tokens/s, Avg generation throughput: 16.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 25.0%
(APIServer pid=1726) INFO 02-19 10:28:11 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 33.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 25.0%
(APIServer pid=1726) INFO:     127.0.0.1:52180 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:28:21 [loggers.py:257] Engine 000: Avg prompt throughput: 369.1 tokens/s, Avg generation throughput: 28.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.8%
(APIServer pid=1726) INFO:     127.0.0.1:38128 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:28:31 [loggers.py:257] Engine 000: Avg prompt throughput: 381.7 tokens/s, Avg generation throughput: 32.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.6%
(APIServer pid=1726) INFO:     127.0.0.1:49664 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:28:41 [loggers.py:257] Engine 000: Avg prompt throughput: 360.7 tokens/s, Avg generation throughput: 38.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 24.7%
(APIServer pid=1726) INFO 02-19 10:28:51 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 38.8 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.7%
(APIServer pid=1726) INFO:     127.0.0.1:45138 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:29:01 [loggers.py:257] Engine 000: Avg prompt throughput: 361.3 tokens/s, Avg generation throughput: 50.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.6%
(APIServer pid=1726) INFO 02-19 10:29:11 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.6%
(APIServer pid=1726) INFO:     127.0.0.1:45748 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:29:21 [loggers.py:257] Engine 000: Avg prompt throughput: 353.0 tokens/s, Avg generation throughput: 35.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 24.5%
(APIServer pid=1726) INFO:     127.0.0.1:47090 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:29:31 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 41.8 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 24.4%
(APIServer pid=1726) INFO 02-19 10:29:41 [loggers.py:257] Engine 000: Avg prompt throughput: 378.5 tokens/s, Avg generation throughput: 55.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.0%, Prefix cache hit rate: 24.4%
(APIServer pid=1726) INFO 02-19 10:29:51 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 20.9 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.4%
(APIServer pid=1726) INFO:     127.0.0.1:54858 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:30:01 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.4%
(APIServer pid=1726) INFO 02-19 10:30:11 [loggers.py:257] Engine 000: Avg prompt throughput: 361.7 tokens/s, Avg generation throughput: 55.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 24.3%
(APIServer pid=1726) INFO 02-19 10:30:21 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 21.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.3%
(APIServer pid=1726) INFO:     127.0.0.1:51076 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:30:31 [loggers.py:257] Engine 000: Avg prompt throughput: 394.8 tokens/s, Avg generation throughput: 37.8 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.0%, Prefix cache hit rate: 24.1%
(APIServer pid=1726) INFO:     127.0.0.1:50338 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=1726) INFO 02-19 10:30:41 [loggers.py:257] Engine 000: Avg prompt throughput: 359.9 tokens/s, Avg generation throughput: 32.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 24.2%
(APIServer pid=1726) INFO 02-19 10:30:51 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 56.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.2%
(APIServer pid=1726) INFO 02-19 10:31:01 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 24.2%
