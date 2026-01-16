# Hybrid Intent Router & Action Flow Engine

Enterprise-grade Chatbot System combining deterministic Rules with semantic Embeddings for high-precision Intent Routing and automated Slot Filling.

## 🌟 Key Features

### 🧠 Hybrid Router V2 (New Architecture)
- **High Performance**: Built on Rust-based `embed-anything` engine, offering 4x speed improvement over V1.
- **Thread-Safe**: Fully thread-safe design with `RWLock` and Atomic State updates.
- **Scalable Storage**: Abstract Vector Store supporting In-Memory, FAISS, and Qdrant backends.
- **Hot Reload**: Zero-downtime config updates using Atomic Config Watcher.
- **Smart Caching**: O(1) TTL Cache for recurring queries.
- **Rule-based Scoring**: Deterministic matching using keyword rules for high precision.
- **Hybrid Fusion**: Weighted combination of Rule + Vector scores for optimal decision making.

### ⚡ Action Flow Engine
- **State Management**: Handles multi-turn conversations (Collecting -> Draft -> Confirmed).
- **Entity Extraction**: Smartly extracts entities (Dates, Numbers, Emails) from natural language.
- **Multi-date Support**: Capable of handling complex date ranges (e.g., "nghỉ từ hôm nay đến ngày mai").
- **Context Handover**: Seamlessly passes context from Router to Action Flow to minimize repetitive questions.

### 💻 Modern Frontend
- **React + Vite**: Fast, responsive SPA.
- **UX Polishing**: Typing indicators, auto-scroll, auto-focus, glassmorphism design.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Rust Compiler (optional, required for compiling `embed-anything` from source)

### 1. Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run Server (V2)
python scripts/run_server.py
or 
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env file (optional, defaults to localhost:8000)
echo "VITE_API_URL=http://localhost:8000" > .env

# Run Dev Server
npm run dev
```

Visit `http://localhost:5173` to interact with the bot.

## 📂 Project Structure

```
├── app/
│   ├── router/             # Hybrid Router Logic (V2 Architecture)
│   │   ├── router_final.py           # V2 Orchestrator
│   │   ├── embed_anything_engine_final.py # Rust-based Engine
│   │   ├── vector_store_final.py     # Storage Abstraction
│   │   └── ...
│   ├── action_flow/        # State Machine & Entity Extractor
│   ├── core/               # Pydantic Models
│   ├── utils/              # Config Loader, Logger
│   └── main.py             # FastAPI Entrypoint (V2)
├── config/
│   ├── action_catalog.yaml # Definitions of Actions & Slots
│   ├── keyword_rules.yaml  # Rule-based matching patterns
│   └── learning_loop.yaml  # Auto-tuning configuration
├── frontend/               # React + Vite Application
├── scripts/                # Utility scripts (Run Server, Benchmark, AutoTuner)
├── tests/                  # Unit & Integration Tests
└── logs/                   # Interaction & Feedback Logs
```

## 🛠 Configuration

### Environment Variables
The system uses `.env` file for configuration. Example:

```env
CHATBOT_SYSTEM__ENV=dev
CHATBOT_SYSTEM__EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHATBOT_SYSTEM__VECTOR_STORE=memory
CHATBOT_LOGGING__LEVEL=INFO
```

### Adding a New Action
Edit `config/action_catalog.yaml`:
```yaml
- action_id: my_new_action
  domain: general
  business_description: "Description of what this action does"
  seed_phrases:
    - "example phrase 1"
    - "example phrase 2"
  required_slots:
    - slot_name_1
  typical_entities:
    - date
```

## 🧪 Testing

### Backend Tests
```bash
# Run all unit tests
python -m unittest discover tests

# Run coverage report
python -m pytest tests/router/ --cov=app.router
```

### Benchmarking
Compare performance metrics:
```bash
python scripts/benchmark_comparison.py
```

## 📚 Documentation
- `docs/ARCHITECTURE.md`: Detailed System Architecture.
- `docs/API_SPEC.md`: API endpoints reference.
- `docs/USER_MANUAL.md`: User guide.

## 📝 License
[MIT](LICENSE)
