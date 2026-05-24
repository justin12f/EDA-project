# Factory architecture

## Layers

1. **Local factories** — per domain and backend (`ReaderFactory`, `DataCleaningStepFactory`, `DataAnalyzerFactory`, `DescriptiveStatisticsFactory`, …).
2. **InyeccionDependency** — binds a backend and delegates to the local factory (`ReadersInyeccionDependency`, `DataCleaningInyeccionDependency`, …).
3. **AgentMasterFactory** — composition root in `agents/master_factory.py`; agents and tools receive one backend for the session.

## Backends

Supported values: `pandas`, `polars`, `spark` (see `core/backend.py`).

## Entry points

```python
from agents.master_factory import AgentMasterFactory

master = AgentMasterFactory("polars")
frame = master.readers().read("data.csv")
step = master.cleaning().create("handle_sentinel_values", frame)
analyzer = master.analyzers().create("shape", frame)
```

## Statistics

- **Descriptive** — native backends in `statistics/descriptive/backends/`.
- **Other domains** — auto-registered via `statistics/domain_registry.py` (pandas calculators wrapped; polars/spark use explicit materialization in `statistics/core/frame_extract.py` until native implementations land).
