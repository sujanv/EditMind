# EditMind High-Performance Gateway & Audit Service

A Golang microservice providing fast-path inference caching, query interception, and knowledge edit audit logging for LLMs.

## Features
- **Sub-millisecond Interception**: Resolves queries matching edited memory before hitting deep neural network layers.
- **Audit Logging**: Immutable ledger of all applied knowledge modifications with timestamps, operators, and latency telemetry.
- **Concurrency**: Thread-safe synchronized storage suitable for production high-throughput workloads.

## Building and Running
```bash
cd gateway
go build -o editmind-gateway main.go
./editmind-gateway
```

## Endpoints
- `GET /health` - System health check.
- `POST /api/gateway/edit` - Register an edited knowledge triple into the fast-path registry.
- `POST /api/gateway/query` - Route a prompt through the knowledge interception engine.
- `GET /api/gateway/audit` - Retrieve the audit ledger of all edits.
