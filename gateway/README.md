# EditMind High-Performance Gateway & Audit Service

A Golang microservice providing sub-millisecond fast-path inference caching, semantic vector similarity routing, token-bucket rate limiting, and tamper-evident audit logging for LLMs.

## Features
- **Sub-Millisecond Interception**: Intercepts queries matching edited memory before hitting deep neural network layers.
- **Semantic Vector Similarity Cache**: Cosine similarity search over vector embeddings with threshold matching ($\ge 0.85$).
- **Token-Bucket Rate Limiter**: Built-in burst handling and request throttling to prevent backend exhaustion.
- **Prometheus Metrics Exporter**: Native `/metrics` endpoint exporting counters and gauges for enterprise observability.
- **Audit Ledger**: Immutable thread-safe ledger recording all applied knowledge modifications with timestamps and telemetry.

## Building and Running
```bash
cd gateway
go build -o editmind-gateway main.go
./editmind-gateway
```

## Running Tests
```bash
cd gateway
go test -v ./...
```

## Endpoints
- `GET /health` - Health check and edit counters.
- `GET /metrics` - Prometheus metrics format.
- `POST /api/gateway/edit` - Register an edited knowledge triple (with optional embedding).
- `POST /api/gateway/query` - Route a query through the exact + semantic cache.
- `GET /api/gateway/audit` - View the complete audit ledger.
