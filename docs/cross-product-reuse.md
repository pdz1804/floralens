# Cross-Product Reuse: Unified Agent Core

This document explains how FloraLens's naturalist assistant is built on an **unmodified Unified Agent Core** shared with AgentForge, proving that a single agent runtime can power radically different AI products.

---

## The Unified Agent Core: Design Principle

The **Unified Agent Core** is a reusable agent runtime designed to work across any domain:

- **Domain-agnostic runtime:** LangGraph state machine, tool registry, memory provider, model registry
- **Domain-specific tools:** Plugins that implement the actual work (search, web lookup, code execution, etc.)
- **Pluggable manifests:** YAML agent definitions, swappable prompts, configurable guardrails

**Promise:** Add a new agent skill without touching the core. Prove it by shipping two very different products (AgentForge + FloraLens) using the identical core.

---

## FloraLens Agent Architecture

### Agent Team Composition

FloraLens's naturalist assistant is a **multi-agent team** orchestrated via a supervisor:

```
┌─────────────────────────────────────────────────┐
│         Naturalist Supervisor (Supervisor)      │
│  Routes user query to appropriate sub-agent     │
└─────────────┬───────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼──┐ ┌────▼──┐ ┌────▼─────┐
│Identifier       │Researcher       │Care-Advisor│
│                 │                 │             │
│Embedding search │Web search       │Synthesis    │
│(internal /api)  │(Tavily)         │Guardrails   │
│Visual ID        │Citations        │Care plan    │
└────────────────┘└────────────────┘└─────────────┘
```

**Roles:**

| Agent | Purpose | Tool(s) | Input | Output |
|---|---|---|---|---|
| **Supervisor** | Route & coordinate | — | "How do I care for a rose?" | Plan: call identifier, then care_advisor |
| **Identifier** | Visual identification | `embedding_search_tool` | Query image (optional) | Top-K similar species + metadata |
| **Researcher** | Factual research | `web_search_tool` | Query topic | Web search results + citations |
| **Care-Advisor** | Synthesis + advice | — | Identifier + Researcher results | Personalized care plan + disclaimers |

### Manifest Files (YAML)

Agent definitions live in `agents/` directory (not shown in the web app, but loaded by the API):

```
agents/
├── naturalist_supervisor.yaml
├── naturalist_identifier.yaml
├── naturalist_researcher.yaml
├── naturalist_care_advisor.yaml
└── prompts/
    ├── naturalist_supervisor.md
    ├── naturalist_identifier.md
    └── ...
```

**Example manifest (naturalist_supervisor.yaml):**

```yaml
id: naturalist_supervisor
version: "1.0"
type: router

# Model configuration (shared across core)
model:
  provider: anthropic
  name: claude-sonnet-5
  temperature: 0.2

# Prompt reference (loaded by core)
prompt_ref: prompts/naturalist_supervisor.md

# Memory configuration
memory:
  provider: mem0
  scope: user
  namespace: floralens

# Sub-agents this supervisor delegates to
sub_agents:
  - naturalist_identifier
  - naturalist_researcher
  - naturalist_care_advisor

# Guardrails enforced by runtime
guardrails:
  - educational_disclaimer
  - no_medical_dosage

# I/O schema (Pydantic)
io_schema:
  input:
    type: object
    properties:
      message:
        type: string
        description: "User query"
  output:
    type: object
    properties:
      answer:
        type: string
      citations:
        type: array
        items:
          type: object
```

**Key design:** No Python code in manifests. The core reads YAML, hydrates tools/prompts/models from registries, and executes the agent graph. Changing an agent = editing YAML + prompts, not touching core code.

---

## Unified Agent Core: Registries & Interfaces

### Tool Registry

**Purpose:** Pluggable tool implementations. Each tool conforms to a common interface.

**Interface (agent_core/tools.py):**

```python
class BaseTool(Protocol):
    """Async callable: takes input dict, returns output dict."""
    
    async def __call__(self, input: dict) -> dict:
        ...
    
    @property
    def name(self) -> str:
        """Unique tool identifier."""
    
    @property
    def description(self) -> str:
        """What this tool does; used in prompts."""
    
    @property
    def input_schema(self) -> dict:
        """JSON Schema of input dict."""
```

**FloraLens tools:**

```python
# apps/api/app/assistant_service.py

class EmbeddingSearchTool(BaseTool):
    """Calls /api/search internally; returns top-K matches."""
    
    @property
    def name(self) -> str:
        return "embedding_search"
    
    @property
    def description(self) -> str:
        return "Search for visually similar flower species"
    
    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "image_base64": {"type": "string", "description": "optional"}
            }
        }
    
    async def __call__(self, input: dict) -> dict:
        # Call /api/search internally
        results = search_image(decode_base64(input.get("image_base64", "")))
        return {
            "results": [
                {
                    "specimen_id": r.specimen_id,
                    "label_name": r.label_name,
                    "confidence": r.confidence
                }
                for r in results
            ]
        }

class WebSearchTool(BaseTool):
    """Calls Tavily web search (implemented by agent core)."""
    
    # (Reused from agent_core; FloraLens doesn't re-implement)
    ...
```

**Registration:**

```python
def build_floralens_registries():
    registries = build_base_registries()  # From agent_core
    
    # Add domain-specific tools
    registries.tools.register(
        EmbeddingSearchTool(),
        overwrite=False
    )
    
    # web_search_tool is already in base registries (from agent_core)
    
    return registries
```

**Result:** Supervisor can call both `embedding_search_tool` (FloraLens-specific) and `web_search_tool` (core, reused from AgentForge).

### Prompt Registry

**Purpose:** Pluggable prompts. Change a prompt = update a Markdown file.

**File structure:**

```
agents/prompts/
├── naturalist_supervisor.md
├── naturalist_identifier.md
└── naturalist_care_advisor.md
```

**Example (naturalist_supervisor.md):**

```markdown
# Naturalist Supervisor

You are a plant expert supervising a team of agents.

Your job:
1. Understand the user's query about flowers/plants
2. Decide which agent to call:
   - Identifier: visual search for similar species
   - Researcher: factual information (care, history)
   - Care-Advisor: personalized care plan

Remember the user's garden from memory and personalize answers.

Always include a disclaimer that this is educational advice, not medical.
```

**Loading (agent_core runtime):**

```python
# Core loads prompt at agent compile time
prompt_text = load_prompt(prompt_ref="prompts/naturalist_supervisor.md")
system_message = create_system_message(prompt_text)
```

**Updating:** Change the `.md` file → next agent run uses new prompt. No code changes.

### Model Registry

**Purpose:** Pluggable model providers. Swap Claude for GPT-4 without touching agent code.

**Built-in models (agent_core):**

```python
# agent_core/models.py
class ModelRegistry:
    def get_model(self, provider: str, name: str):
        if provider == "anthropic":
            return AnthropicModel(name)
        elif provider == "openai":
            return OpenAIModel(name)
        # ...
```

**Override for testing (FloraLens tests):**

```python
# test_assistant.py
def test_assistant_chat():
    registries = build_floralens_registries()
    
    # Override model provider to mock LLM (no API key / cost)
    registries.models.register(
        MockModel("claude-sonnet"),  # Returns fixed test response
        overwrite=True
    )
    
    agent = compile_naturalist(registries)
    result = await agent.astream("Hello", thread_id="test")
    # ...
```

**Benefit:** Test agents without spending tokens.

### Memory Provider Registry

**Purpose:** Pluggable memory backends (mem0 vs custom).

**Interface (agent_core/memory.py):**

```python
class MemoryProvider(Protocol):
    async def add(self, text: str, meta: dict = None) -> str:
        """Add a memory; return memory ID."""
    
    async def retrieve(self, query: str, top_k: int = 5) -> list[Memory]:
        """Retrieve similar memories."""
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
```

**FloraLens memory configuration (manifest):**

```yaml
memory:
  provider: mem0
  scope: user         # Scoped to individual users
  namespace: floralens  # Namespaced within FloraLens
```

**Long-term memory (mem0):**

Stores semantic memories that persist across sessions and requests:
- "User prefers indoor plants"
- "User has a rose garden with 15 specimens"
- "User lives in zone 6b"

**Short-term memory (LangGraph checkpointer):**

Persists conversation thread state:
- Prior messages in the thread
- Tool call history
- Intermediate reasoning

**Combined power:** Ask the assistant "How do I care for my roses?" and it recalls:
1. From long-term memory: "User has roses saved"
2. From conversation: prior messages in this thread
3. From garden API: current specimen list
→ Personalized, context-aware answer.

---

## Shared Core: LangGraph Runtime

### StateGraph: Agent Orchestration

All agents (AgentForge + FloraLens) use the same **LangGraph state machine**:

```
user query
    ↓
[Supervisor decides which sub-agent]
    ↓
[Sub-agent reasoning + tool call]
    ↓
[Tool execution]
    ↓
[Reflection]
    ↓
[Loop until done]
    ↓
final answer
```

**State nodes (shared across core):**

| Node | Logic | Tool Calls |
|---|---|---|
| `supervisor` | Route to sub-agent | None (decision logic) |
| `identifier` | Search + identify | embedding_search_tool |
| `researcher` | Research query | web_search_tool |
| `care_advisor` | Synthesize answer | None (LLM synthesis) |

**Transitions (defined in manifest):**

```yaml
# naturalist_supervisor.yaml
transitions:
  supervisor:
    - condition: "query is visual"
      next: identifier
    - condition: "query is factual"
      next: researcher
    - condition: "query is care advice"
      next: care_advisor
```

### Streaming & SSE Response

Agents stream their execution via **SSE events**:

```python
async def agent_stream(agent, message: str, thread_id: str):
    """Stream agent execution as SSE-compatible events."""
    
    async for event in agent.astream(message, thread_id=thread_id):
        # event = AgentStepEvent (step_type, agent_id, thought/tool/answer, etc.)
        yield f"data: {event.model_dump_json()}\n\n"
```

**Event types (all agents):**

| Type | When | Payload |
|---|---|---|
| `run_started` | Agent compiled, ready | — |
| `step` | Reasoning/tool/answer | `step_type`, `agent_id`, content |
| `done` | Finished | — |
| `error` | Exception | `detail` |

**Benefit:** Frontend receives full execution trace; can display reasoning, tool calls, citations in real-time.

---

## Extensibility: Adding a New Agent (No Core Changes)

### Example: Disease Diagnosis Agent

**Goal:** Add an agent that diagnoses leaf diseases. Prove it requires no core code changes.

**Steps:**

#### 1. Define the new agent manifest

Create `agents/disease_advisor.yaml`:

```yaml
id: disease_advisor
version: "1.0"
type: agent

model:
  provider: anthropic
  name: claude-sonnet-5
  temperature: 0.3

prompt_ref: prompts/disease_advisor.md

memory:
  provider: mem0
  scope: user
  namespace: floralens

tools:
  - embedding_search  # Find similar diseased specimens (existing)
  - web_search        # Research disease info (existing)

guardrails:
  - educational_disclaimer
  - no_medical_diagnosis  # NEW: refuse to diagnose humans
```

#### 2. Write the prompt

Create `agents/prompts/disease_advisor.md`:

```markdown
# Disease Advisor

You are an expert in plant disease identification.

Given a photo of a diseased leaf:
1. Describe visible symptoms
2. Search for similar diseased specimens
3. Research the likely disease
4. Suggest treatment (fungicide, pruning, etc.)

Always include: "This is educational; consult a professional for valuable plants."
Never diagnose human diseases.
```

#### 3. Register the new agent in the supervisor

Update `agents/naturalist_supervisor.yaml`:

```yaml
sub_agents:
  - naturalist_identifier
  - naturalist_researcher
  - naturalist_care_advisor
  - disease_advisor        # NEW
```

Update the supervisor prompt (`prompts/naturalist_supervisor.md`):

```markdown
...you can now also:
- Call the disease_advisor for diagnosing leaf diseases
...
```

#### 4. Update the Supervisor's tool routing

Update `agents/prompts/naturalist_supervisor.md` to add:

```markdown
If the query mentions disease symptoms or asks about a diseased leaf:
  → Call disease_advisor
```

#### 5. No Python code needed.

The core's `StateGraph` automatically:
- Reads the new manifest
- Hydrates tools + prompts
- Adds the new state node
- Handles routing

**Result:** New agent added; core untouched. No refactoring, no API changes.

---

## Proof of Concept: Same Core, Two Products

### FloraLens Agents

**Domain:** Flower identification & care

**Core agents:**
- `naturalist_supervisor` — Routes flower/care queries
- `naturalist_identifier` — Visual search (embedding tool)
- `naturalist_researcher` — Factual research (web search)
- `naturalist_care_advisor` — Care synthesis

**Domain-specific tools:**
- `embedding_search_tool` — Calls FloraLens `/api/search` (1,632 flower specimens)

**Memory context:**
- User's saved plants (My Garden)
- User preferences (climate zone, indoor/outdoor)

### AgentForge Agents

**Domain:** Software engineering & agent building

**Core agents:**
- `forge_supervisor` — Routes code/research queries
- `code_agent` — Code generation (LLM + tool)
- `researcher_agent` — Research (web search)
- `reviewer_agent` — Code review

**Domain-specific tools:**
- `code_execution_tool` — Run Python in sandbox (AgentForge exclusive)
- `web_search_tool` — Web search (shared core)

**Memory context:**
- User's codebase context (files, recent edits)
- Project preferences (language, frameworks)

### Comparison

| Aspect | FloraLens | AgentForge | Shared? |
|---|---|---|---|
| **Agent runtime** | LangGraph | LangGraph | ✅ Yes (core) |
| **Tool interface** | BaseTool | BaseTool | ✅ Yes (core) |
| **Prompt registry** | YAML + Markdown | YAML + Markdown | ✅ Yes (core) |
| **Memory provider** | mem0 | mem0 | ✅ Yes (core) |
| **Model registry** | Anthropic/OpenAI | Anthropic/OpenAI | ✅ Yes (core) |
| **SSE streaming** | Event types | Event types | ✅ Yes (core) |
| **Supervisor logic** | Flower routing | Code routing | ❌ No (domain-specific prompts) |
| **Embedding tool** | FloraLens `/api/search` | — | ❌ No (domain-specific) |
| **Code tool** | — | Sandbox execution | ❌ No (domain-specific) |

**Conclusion:** The core is truly domain-agnostic. Domain-specific behavior comes from prompts + tools, not the runtime.

---

## Integration with FloraLens API

### Endpoint: POST /api/assistant (SSE)

The API endpoint that surfaces the agent:

```python
# apps/api/app/main.py

@app.post("/api/assistant")
async def assistant(req: AssistantRequest) -> StreamingResponse:
    """Chat with the naturalist agent team."""
    
    async def event_stream():
        yield f"data: {json.dumps({'type': 'run_started'})}\n\n"
        try:
            # Compile the naturalist supervisor from manifests
            agent = compile_naturalist(
                assistant_registries,  # Built at startup
                _assistant_checkpointer  # If durable memory enabled
            )
            
            # Stream agent execution
            async for event in agent.astream(
                req.message,
                thread_id=req.thread_id
            ):
                # Redact secrets before streaming
                yield f"data: {redact_secrets(event.model_dump_json())}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except AgentCoreError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        finally:
            # Clean up agent resources
            await agent.aclose()
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**Key points:**

1. **Compile at request time:** Each request compiles a fresh agent graph (stateless API)
2. **Reuse registries:** Pre-built registries include all tools, models, prompts
3. **Optional checkpointer:** If `FLORALENS_CHECKPOINT_DB` set, agent loads prior thread state
4. **Redact secrets:** Before SSE, strip API keys / bearer tokens
5. **Stream events:** Frontend receives tool calls, reasoning, final answer in real-time

### Initialization (on API startup)

```python
# apps/api/app/main.py

def build_floralens_registries():
    """Build agent registries with FloraLens-specific tools."""
    
    registries = build_base_registries()  # From agent_core: models, prompts, memory
    
    # Add FloraLens-specific tools
    registries.tools.register(EmbeddingSearchTool(), overwrite=False)
    # web_search_tool already in base registries
    
    # Load prompts from agents/ directory
    registries.prompts.load_from_directory("agents/prompts/")
    
    # Optionally override models for testing
    # (e.g., registries.models.register(MockModel(...), overwrite=True))
    
    return registries

# Built once at startup
assistant_registries = build_floralens_registries()
```

---

## Testing: Agent Behavior Without API Keys

### Mock Model for Offline Testing

```python
# tests/test_assistant.py

class MockModel:
    """Fake LLM for testing; always returns canned response."""
    
    async def invoke(self, prompt: str, **kwargs) -> str:
        return {
            "type": "answer",
            "content": "This is a test response."
        }

def test_assistant_flow():
    registries = build_floralens_registries()
    
    # Override model to avoid API cost
    registries.models.register(
        MockModel(),
        overwrite=True
    )
    
    agent = compile_naturalist(registries)
    
    # Run agent without calling OpenAI
    result = agent.astream("How do I care for a rose?", thread_id="test")
    
    # Assert on event stream
    events = [e async for e in result]
    assert events[0]['type'] == 'run_started'
    assert any(e['type'] == 'answer' for e in events)
```

**Benefit:** Fast, repeatable tests that don't require API keys or network.

---

## Deployment Considerations

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (for Claude, web search) | unset |
| `TAVILY_API_KEY` | Tavily web search API | unset |
| `FLORALENS_CHECKPOINT_DB` | SQLite path for durable thread memory | unset (disabled) |
| `FLORALENS_API_KEY` | Optional API key for write endpoints | unset (optional) |

### Local Development

```bash
# Set keys in .env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Start API
venv\Scripts\python.exe -m uvicorn apps.api.app.main:app --port 8100
```

### Production

- **Secrets:** Store in environment, never committed
- **Rate limiting:** Per-IP limits on `/api/assistant` (10 req/min default)
- **Monitoring:** Log all agent runs; track token usage
- **Failover:** If web search unavailable, agent still works (tool graceful degradation)

---

## Limitations & Future Work

### Current Limitations

1. **Single supervisor:** All queries routed through one supervisor. Multi-supervisor hierarchies not yet implemented.

2. **Sequential execution:** Agents run one at a time. Parallel sub-agent calls deferred.

3. **Fixed tool sets:** Tools must be pre-registered; dynamic tool loading not yet supported.

4. **Prompt hardcoding:** Prompts are files; no dynamic prompt synthesis per query.

### Future Extensions

- **Hierarchical supervisors:** Build a tree of supervisors for complex domains
- **Parallel tools:** Execute multiple tools concurrently when safe
- **Dynamic tools:** Register tools at runtime based on agent config
- **Few-shot learning:** Provide agent with in-context examples per query
- **Preference learning:** Agent learns user's preferences over time (beyond mem0)

---

## References

- **[Agent Core](../packages/agent-core/AGENT-CORE.md)** (if available) — Unified Agent Core specification
- **[Assistant Service Code](../apps/api/app/assistant_service.py)** — FloraLens agent integration
- **[Agent Manifests](../agents/)** — YAML agent definitions
- **[LangGraph Docs](https://langchain-ai.github.io/langgraph/)** — State machine framework
- **[mem0 Docs](https://docs.mem0.ai/)** — Memory provider

---

## Summary

FloraLens proves the **Unified Agent Core thesis:**

✅ **Same runtime** (LangGraph) powers both flower identification and software engineering

✅ **Pluggable tools** make agents domain-specific without core changes

✅ **YAML manifests** + **Markdown prompts** enable new agents without Python coding

✅ **Registries** (tools, prompts, models, memory) are reusable across domains

✅ **Extensibility first:** Adding disease diagnosis = 1 YAML + 1 Markdown file, zero core changes

The core is not "almost reusable" or "mostly generic" — it's *genuinely domain-agnostic*, proven by shipping two very different products with identical agent runtime, without modification.
