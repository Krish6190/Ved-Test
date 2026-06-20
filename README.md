ved — Self-contained minimal chatbot package

Run with:

python -m ved

Structure inside this package:
- `Modelfile.standard` and `Modelfile.turbo` — per-mode model configuration files
- `chatbot.py` — Chatbot + ModelAdapter that reads Modelfile files from this package
- `models/loader.py` — placeholder model loader (`load_model_stub`) to be replaced with real loaders
- `graph/` — placeholder graph integration module
- `gui.py` — Tkinter GUI and entrypoint

Notes:
- The package is intentionally self-contained so you can move it as a standalone project.
- Currently `ModelAdapter` and `models.loader` are stubs that simulate behavior. I can wire in real model loading or LangGraph integration next if you want.
