| Model | Device | Mode | N | Schema % | Gate % | Correct % | TTFT p50 (s) | Decode tok/s | Memory (MB) |
|---|---|---|---|---|---|---|---|---|---|
| llama3.2-3b-instruct-q4_K_M | cpu (llama.cpp) | constrained | 20 | 100 | 65 | 65 | 0.263 | 10.8 | 3671 |
| llama3.2-3b-instruct-q4_K_M | cpu (llama.cpp) | unconstrained | 20 | 100 | 65 | 65 | 0.244 | 20.8 | 3663 |
| llama3.2-3b-instruct-q4_K_M | gpu (ollama) | constrained | 20 | 100 | 60 | 60 | 0.045 | 118.2 | 2436 |
| llama3.2-3b-instruct-q4_K_M | gpu (ollama) | unconstrained | 20 | 100 | 60 | 60 | 0.048 | 104.5 | 2436 |
| llama3.2-3b-instruct-q8_0 | cpu (llama.cpp) | constrained | 20 | 100 | 60 | 60 | 0.562 | 8.7 | 3706 |
| llama3.2-3b-instruct-q8_0 | cpu (llama.cpp) | unconstrained | 20 | 75 | 60 | 60 | 0.554 | 14.0 | 3697 |
| llama3.2-3b-instruct-q8_0 | gpu (ollama) | constrained | 20 | 100 | 50 | 50 | 0.066 | 47.6 | 3774 |
| llama3.2-3b-instruct-q8_0 | gpu (ollama) | unconstrained | 20 | 75 | 50 | 50 | 0.110 | 24.8 | 3774 |
| mistral-7b-instruct-v0.3-q4_K_M | cpu (llama.cpp) | constrained | 20 | 65 | 0 | 0 | 0.663 | 9.2 | 7697 |
| mistral-7b-instruct-v0.3-q4_K_M | cpu (llama.cpp) | unconstrained | 20 | 0 | 0 | 0 | 0.679 | 9.6 | 7671 |
| mistral-7b-instruct-v0.3-q4_K_M | gpu (ollama) | constrained | 20 | 100 | 95 | 95 | 0.188 | 10.8 | 4722 |
| mistral-7b-instruct-v0.3-q4_K_M | gpu (ollama) | unconstrained | 20 | 95 | 90 | 90 | 0.243 | 7.1 | 4722 |
| qwen2.5-coder-7b-instruct-q4_K_M | cpu (llama.cpp) | constrained | 20 | 100 | 100 | 100 | 0.639 | 6.3 | 7511 |
| qwen2.5-coder-7b-instruct-q4_K_M | cpu (llama.cpp) | unconstrained | 20 | 60 | 60 | 60 | 0.620 | 10.0 | 7503 |
| qwen2.5-coder-7b-instruct-q4_K_M | gpu (ollama) | constrained | 20 | 100 | 100 | 100 | 0.172 | 12.3 | 4528 |
| qwen2.5-coder-7b-instruct-q4_K_M | gpu (ollama) | unconstrained | 20 | 65 | 65 | 65 | 0.315 | 6.2 | 4528 |
