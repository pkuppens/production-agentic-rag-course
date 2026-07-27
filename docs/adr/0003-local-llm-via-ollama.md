# Local LLM (Ollama) instead of a cloud LLM API

Generation runs against a local Ollama server (`llama3.2`, `OLLAMA_HOST`/`OLLAMA_MODEL`) rather than a hosted LLM API. This keeps the course free to run — the README states ~$0 local cost with optional cloud APIs only for extras — and guarantees paper content and queries never leave the machine. The trade-off is slower and lower-quality generation than a hosted frontier model; Week 5 mitigates this with an 80% prompt-size reduction for a 6x speedup rather than switching providers.
