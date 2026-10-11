# Model Relay

Chain local Ollama models into a relay: a frontier model (Claude, ChatGPT) or a person
hands a task down through progressively smaller models on one 16 GB Mac, and each layer
checks the answer on the way back up.

```
Frontier → 8–14B → 3–4B → 1B → (answer) → back up, checked at each layer
```

The point is the plumbing and the limits, not solving problems.

## Run

Needs the Ollama app running and the models pulled. No Python packages.

```bash
python3 relay.py "2+2"
python3 relay.py --chain gemma4:12b,gemma3:4b,gemma3:1b "2+2"
python3 relay.py --json "2+2"
python3 relay.py --blind "How many r's are in strawberry?"   # every model answers on its own
python3 relay.py --rule "Answer with just the number." "2+2"  # a rule every model must follow
```

To watch it run, start the web page and open http://localhost:8765:

```bash
python3 server.py
```

Every run, from the page or the command line, is saved as one JSON file in `runs/` (git-ignored):
the prompt, rule, mode, chain, every message each model sent, timings, and the final answer.
The page's **Saved runs** list opens any of them again. Add `--no-save` to skip saving.

## Findings so far (2026-10-10, M5 Air 16 GB)

- Default chain `granite4.2:8b → gemma3:4b → granite4.2:3b → gemma3:1b`: "2+2" in ~13 s.
- Ollama kept at most 3 of the 4 models loaded at once (its default `OLLAMA_MAX_LOADED_MODELS`
  is 3), so a layer gets reloaded on the way back up.
- Forcing JSON output on `gemma3:1b` sent it into a loop; the bottom layer replies in plain
  text, and every reply is capped at 200 tokens.
- `gemma4:e2b` / `e4b` are 7.5 / 9.5 GB on disk, so they're not light layers.
