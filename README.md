# Check-in Buddy

Check-in Buddy is a local FastAPI wellness companion with a senior check-in page and a family activity view. It records user-entered routines; it is not a medical service or emergency monitoring system.

## Run on Windows

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`. Create a free NVIDIA Developer account and generate an NVIDIA NIM API key from [NVIDIA Build](https://build.nvidia.com/). Set `NVIDIA_API_KEY` in `.env`. The app uses the hosted `nvidia/nemotron-3-nano-30b-a3b` model by default. A key is required even when using the free trial; it stays on the server and is never sent to the browser. The model endpoint and trial terms can change; check NVIDIA Build for current availability and quotas.

3. Train the local conversational-tone classifier on both bundled datasets:

   ```powershell
   python -m app.train_emotion_model
   ```

   Training evaluates a held-out split and saves the model to `trained_models/emotion_classifier.joblib`. It samples up to 20,000 examples per label from `text.csv.zip` and 6,000 per label from `go_emotions_dataset.csv.zip`. The trained classifier is used as tentative tone context for the NIM chatbot; it does not generate chat responses, diagnose conditions, or fine-tune NVIDIA's hosted model. The hosted NIM model remains responsible for generative chat.

4. Start the server: 

   ```powershell
   python -m uvicorn app.main:app --reload
   ```

5. Open <http://127.0.0.1:8000>. The first visit creates the demo persona if the database is empty. The family dashboard is at <http://127.0.0.1:8000/family>.

For local UI/API work without an NIM key, set `LLM_PROVIDER=deterministic` in `.env`. This test provider is not a generative assistant. Direct routine logging and the family dashboard work independently of the LLM provider.

## Included flows

- NVIDIA NIM chat, persisted recent conversation history, optional browser speech input, spoken replies, and accessible larger text.
- Dataset-trained seven-class emotion classifier built from the supplied GoEmotions and six-emotion CSV archives.
- Daily check-in, medication, meal, mood, and reminder logging.
- Family activity overview and alert acknowledgement.
- Deterministic safety checks and escalation rules, independent of model output.
- Cited general wellness tips from `docs/care_tips.md`.
- Provider and classifier status at `/api/status`; service health at `/health`.

The supplied emotion datasets contain social-media text and may be noisy or biased. The text classifier is deliberately treated as uncertain tone guidance only. It is not shown as a diagnosis or an objective assessment of a person.

## Tests

Run `python -m pytest`. Tests should use an isolated test database through `DATABASE_URL`; do not point test runs at a database containing data you want to keep.
 