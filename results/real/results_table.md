| Model | Device | Mode | Schema % | Correct % (strict gate) | Correct % if keyword args accepted | TTFT p50 (s) | Decode tok/s | GPU clock, median MHz | Memory (MB) |
|---|---|---|---|---|---|---|---|---|---|
| llama3.2-3b-instruct-q4_K_M | cpu (llama.cpp) | constrained | 100 | 65 | 100 | 0.263 | 10.8 | - | 3671 |
| llama3.2-3b-instruct-q4_K_M | cpu (llama.cpp) | unconstrained | 100 | 65 | 100 | 0.244 | 20.8 | - | 3663 |
| llama3.2-3b-instruct-q4_K_M | gpu (ollama) | constrained | 100 | 60 | 95 | 0.056 | 126.4 | 1200 | 2436 |
| llama3.2-3b-instruct-q4_K_M | gpu (ollama) | unconstrained | 100 | 60 | 95 | 0.057 | 134.5 | 1095 | 2436 |
| llama3.2-3b-instruct-q8_0 | cpu (llama.cpp) | constrained | 100 | 60 | 70 | 0.562 | 8.7 | - | 3706 |
| llama3.2-3b-instruct-q8_0 | cpu (llama.cpp) | unconstrained | 75 | 60 | 70 | 0.554 | 14.0 | - | 3697 |
| llama3.2-3b-instruct-q8_0 | gpu (ollama) | constrained | 100 | 50 | 65 | 0.054 | 101.8 | 1170 | 3774 |
| llama3.2-3b-instruct-q8_0 | gpu (ollama) | unconstrained | 75 | 50 | 65 | 0.061 | 103.8 | 1418 | 3774 |
| mistral-7b-instruct-v0.3-q4_K_M | cpu (llama.cpp) | constrained | 100 | 95 | 95 | 0.629 | 8.2 | - | 7696 |
| mistral-7b-instruct-v0.3-q4_K_M | cpu (llama.cpp) | unconstrained | 95 | 90 | 90 | 0.698 | 8.8 | - | 7690 |
| mistral-7b-instruct-v0.3-q4_K_M | gpu (ollama) | constrained | 100 | 95 | 95 | 0.068 | 64.2 | 952 | 4722 |
| mistral-7b-instruct-v0.3-q4_K_M | gpu (ollama) | unconstrained | 95 | 90 | 90 | 0.059 | 63.6 | 945 | 4722 |
| qwen2.5-coder-7b-instruct-q4_K_M | cpu (llama.cpp) | constrained | 100 | 100 | 100 | 0.639 | 6.3 | - | 7511 |
| qwen2.5-coder-7b-instruct-q4_K_M | cpu (llama.cpp) | unconstrained | 60 | 60 | 60 | 0.620 | 10.0 | - | 7503 |
| qwen2.5-coder-7b-instruct-q4_K_M | gpu (ollama) | constrained | 100 | 100 | 100 | 0.067 | 61.4 | 840 | 4528 |
| qwen2.5-coder-7b-instruct-q4_K_M | gpu (ollama) | unconstrained | 65 | 65 | 65 | 0.069 | 62.6 | 878 | 4528 |
