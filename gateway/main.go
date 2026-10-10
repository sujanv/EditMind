// EditMind High-Performance Knowledge Gateway & Audit Service
// Features: Vector Similarity Cache, Token-Bucket Rate Limiter, Prometheus Metrics, & Audit Trail.

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
)

type KnowledgeEdit struct {
	ID        string    `json:"id"`
	Subject   string    `json:"subject"`
	Relation  string    `json:"relation"`
	Target    string    `json:"target"`
	OldTarget string    `json:"old_target,omitempty"`
	Editor    string    `json:"editor"`
	Embedding []float64 `json:"embedding,omitempty"`
	Timestamp time.Time `json:"timestamp"`
	LatencyMs float64   `json:"latency_ms"`
}

type QueryRequest struct {
	Prompt    string    `json:"prompt"`
	Embedding []float64 `json:"embedding,omitempty"`
}

type QueryResponse struct {
	Prompt       string  `json:"prompt"`
	Predicted    string  `json:"predicted"`
	Intercepted  bool    `json:"intercepted"`
	Similarity   float64 `json:"similarity"`
	MatchedEdit  string  `json:"matched_edit,omitempty"`
	ResponseTime string  `json:"response_time"`
}

type TokenBucket struct {
	capacity   float64
	tokens     float64
	refillRate float64
	lastRefill time.Time
	mu         sync.Mutex
}

func NewTokenBucket(capacity float64, refillRate float64) *TokenBucket {
	return &TokenBucket{
		capacity:   capacity,
		tokens:     capacity,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

func (tb *TokenBucket) Allow() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.tokens = math.Min(tb.capacity, tb.tokens+elapsed*tb.refillRate)
	tb.lastRefill = now

	if tb.tokens >= 1.0 {
		tb.tokens -= 1.0
		return true
	}
	return false
}

type GatewayServer struct {
	mu          sync.RWMutex
	edits       map[string]KnowledgeEdit
	auditLogs   []KnowledgeEdit
	rateLimiter *TokenBucket

	// Metrics counters
	totalRequests     int64
	interceptedCount  int64
	rateLimitedCount  int64
}

func NewGatewayServer() *GatewayServer {
	return &GatewayServer{
		edits:       make(map[string]KnowledgeEdit),
		auditLogs:   make([]KnowledgeEdit, 0),
		rateLimiter: NewTokenBucket(100.0, 50.0), // 100 capacity, 50 req/sec refill
	}
}

func CosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0.0
	}
	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0.0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func (s *GatewayServer) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "ok",
		"service":     "EditMind Go Gateway",
		"version":     "0.2.0",
		"total_edits": len(s.edits),
	})
}

func (s *GatewayServer) HandleMetrics(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# HELP editmind_gateway_requests_total Total HTTP requests handled.\n")
	fmt.Fprintf(w, "# TYPE editmind_gateway_requests_total counter\n")
	fmt.Fprintf(w, "editmind_gateway_requests_total %d\n", s.totalRequests)

	fmt.Fprintf(w, "# HELP editmind_gateway_intercepted_total Total queries intercepted by fast-path cache.\n")
	fmt.Fprintf(w, "# TYPE editmind_gateway_intercepted_total counter\n")
	fmt.Fprintf(w, "editmind_gateway_intercepted_total %d\n", s.interceptedCount)

	fmt.Fprintf(w, "# HELP editmind_gateway_active_edits Number of active knowledge edits in memory.\n")
	fmt.Fprintf(w, "# TYPE editmind_gateway_active_edits gauge\n")
	fmt.Fprintf(w, "editmind_gateway_active_edits %d\n", len(s.edits))
}

func (s *GatewayServer) HandleRegisterEdit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var edit KnowledgeEdit
	if err := json.NewDecoder(r.Body).Decode(&edit); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	edit.Timestamp = time.Now()
	key := strings.ToLower(strings.TrimSpace(edit.Subject))

	s.mu.Lock()
	s.edits[key] = edit
	s.auditLogs = append(s.auditLogs, edit)
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "registered",
		"edit":   edit,
	})
}

func (s *GatewayServer) HandleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if !s.rateLimiter.Allow() {
		s.mu.Lock()
		s.rateLimitedCount++
		s.mu.Unlock()
		http.Error(w, "Rate limit exceeded (token bucket exhausted)", http.StatusTooManyRequests)
		return
	}

	start := time.Now()
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	s.totalRequests++
	s.mu.Unlock()

	lowerPrompt := strings.ToLower(req.Prompt)
	s.mu.RLock()
	defer s.mu.RUnlock()

	intercepted := false
	predicted := "unknown"
	matchedKey := ""
	bestSim := 0.0

	// 1. Exact / Substring search
	for subj, edit := range s.edits {
		if strings.Contains(lowerPrompt, subj) {
			intercepted = true
			predicted = edit.Target
			matchedKey = fmt.Sprintf("%s (%s -> %s)", edit.Subject, edit.Editor, edit.Target)
			bestSim = 1.0
			break
		}
	}

	// 2. Vector Cosine Similarity Fallback if embedding provided
	if !intercepted && len(req.Embedding) > 0 {
		for _, edit := range s.edits {
			if len(edit.Embedding) > 0 {
				sim := CosineSimilarity(req.Embedding, edit.Embedding)
				if sim > 0.85 && sim > bestSim {
					bestSim = sim
					intercepted = true
					predicted = edit.Target
					matchedKey = fmt.Sprintf("semantic_match::%s (sim: %.3f)", edit.Subject, sim)
				}
			}
		}
	}

	if intercepted {
		s.mu.RUnlock()
		s.mu.Lock()
		s.interceptedCount++
		s.mu.Unlock()
		s.mu.RLock()
	}

	resp := QueryResponse{
		Prompt:       req.Prompt,
		Predicted:    predicted,
		Intercepted:  intercepted,
		Similarity:   bestSim,
		MatchedEdit:  matchedKey,
		ResponseTime: time.Since(start).String(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *GatewayServer) HandleAudit(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"total_count": len(s.auditLogs),
		"audit_trail": s.auditLogs,
	})
}

func main() {
	server := NewGatewayServer()

	http.HandleFunc("/health", server.HandleHealth)
	http.HandleFunc("/metrics", server.HandleMetrics)
	http.HandleFunc("/api/gateway/edit", server.HandleRegisterEdit)
	http.HandleFunc("/api/gateway/query", server.HandleQuery)
	http.HandleFunc("/api/gateway/audit", server.HandleAudit)

	port := ":8080"
	log.Printf("[EditMind] Enhanced Gateway microservice running on http://localhost%s\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
