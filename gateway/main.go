// EditMind High-Performance Knowledge Gateway & Audit Service
// Features: Vector Similarity Cache, Token-Bucket Rate Limiter, Prometheus Metrics, & Audit Trail.

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
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
	walPath     string
	walMu       sync.Mutex

	// Metrics and latency counters
	totalRequests     int64
	interceptedCount  int64
	rateLimitedCount  int64
	latencies         []float64
}

func NewGatewayServer(walPath ...string) *GatewayServer {
	path := ""
	if len(walPath) > 0 {
		path = walPath[0]
	}
	s := &GatewayServer{
		edits:       make(map[string]KnowledgeEdit),
		auditLogs:   make([]KnowledgeEdit, 0),
		rateLimiter: NewTokenBucket(100.0, 50.0), // 100 capacity, 50 req/sec refill
		walPath:     path,
		latencies:   make([]float64, 0, 1000),
	}
	if path != "" {
		_ = s.ReplayWAL()
	}
	return s
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
	_ = s.AppendWAL(edit)

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

	elapsedMs := float64(time.Since(start).Microseconds()) / 1000.0
	s.mu.RUnlock()
	s.mu.Lock()
	s.latencies = append(s.latencies, elapsedMs)
	if len(s.latencies) > 5000 {
		s.latencies = s.latencies[1000:]
	}
	s.mu.Unlock()
	s.mu.RLock()

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

type MultiHopChainRequest struct {
	AnchorSubject string `json:"anchor_subject"`
	HopRelation   string `json:"hop_relation"`
}

type MultiHopChainResponse struct {
	ResolvedTarget string   `json:"resolved_target"`
	TraversalPath  []string `json:"traversal_path"`
	HopsCount      int      `json:"hops_count"`
	Found          bool     `json:"found"`
}

func (s *GatewayServer) ResolveMultiHop(anchor string, relation string) (string, []string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	key := strings.ToLower(strings.TrimSpace(anchor))
	firstEdit, exists := s.edits[key]
	if !exists {
		return "", nil, false
	}

	path := []string{firstEdit.Subject, firstEdit.Target}
	targetKey := strings.ToLower(strings.TrimSpace(firstEdit.Target))
	if secondEdit, exists2 := s.edits[targetKey]; exists2 {
		path = append(path, secondEdit.Target)
		return secondEdit.Target, path, true
	}

	return firstEdit.Target, path, true
}

func (s *GatewayServer) HandleMultiHopResolve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req MultiHopChainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	target, path, found := s.ResolveMultiHop(req.AnchorSubject, req.HopRelation)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(MultiHopChainResponse{
		ResolvedTarget: target,
		TraversalPath:  path,
		HopsCount:      len(path) - 1,
		Found:          found,
	})
}

func (s *GatewayServer) AppendWAL(edit KnowledgeEdit) error {
	if s.walPath == "" {
		return nil
	}
	s.walMu.Lock()
	defer s.walMu.Unlock()

	f, err := os.OpenFile(s.walPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	bytes, err := json.Marshal(edit)
	if err != nil {
		return err
	}
	_, err = f.WriteString(string(bytes) + "\n")
	return err
}

func (s *GatewayServer) ReplayWAL() error {
	if s.walPath == "" {
		return nil
	}
	s.walMu.Lock()
	defer s.walMu.Unlock()

	f, err := os.Open(s.walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	s.mu.Lock()
	defer s.mu.Unlock()

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		var edit KnowledgeEdit
		if err := json.Unmarshal([]byte(line), &edit); err == nil {
			key := strings.ToLower(strings.TrimSpace(edit.Subject))
			s.edits[key] = edit
			s.auditLogs = append(s.auditLogs, edit)
		}
	}
	return scanner.Err()
}

func (s *GatewayServer) CalculatePercentiles() (p50, p90, p99 float64) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.latencies) == 0 {
		return 0, 0, 0
	}
	copied := make([]float64, len(s.latencies))
	copy(copied, s.latencies)
	sort.Float64s(copied)

	n := len(copied)
	p50 = copied[int(float64(n)*0.50)]
	p90 = copied[int(float64(n)*0.90)]
	p99 = copied[int(float64(n)*0.99)]
	return p50, p90, p99
}

func (s *GatewayServer) HandleStats(w http.ResponseWriter, r *http.Request) {
	p50, p90, p99 := s.CalculatePercentiles()
	s.mu.RLock()
	defer s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"total_requests":     s.totalRequests,
		"intercepted_count":  s.interceptedCount,
		"rate_limited_count": s.rateLimitedCount,
		"active_edits_count": len(s.edits),
		"latency_p50_ms":     p50,
		"latency_p90_ms":     p90,
		"latency_p99_ms":     p99,
	})
}

func (s *GatewayServer) HandleSnapshotExport(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"active_edits": len(s.edits),
		"edits":        s.edits,
		"audit_trail":  s.auditLogs,
		"exported_at":  time.Now(),
	})
}

func (s *GatewayServer) HandleSnapshotImport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var payload struct {
		Edits      map[string]KnowledgeEdit `json:"edits"`
		AuditTrail []KnowledgeEdit          `json:"audit_trail"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for k, v := range payload.Edits {
		s.edits[k] = v
		_ = s.AppendWAL(v)
	}
	s.auditLogs = append(s.auditLogs, payload.AuditTrail...)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":         "imported",
		"imported_edits": len(payload.Edits),
		"total_edits":    len(s.edits),
	})
}

func main() {
	server := NewGatewayServer("gateway_wal.log")

	http.HandleFunc("/health", server.HandleHealth)
	http.HandleFunc("/metrics", server.HandleMetrics)
	http.HandleFunc("/api/gateway/edit", server.HandleRegisterEdit)
	http.HandleFunc("/api/gateway/query", server.HandleQuery)
	http.HandleFunc("/api/gateway/multihop", server.HandleMultiHopResolve)
	http.HandleFunc("/api/gateway/audit", server.HandleAudit)
	http.HandleFunc("/api/gateway/stats", server.HandleStats)
	http.HandleFunc("/api/gateway/snapshot/export", server.HandleSnapshotExport)
	http.HandleFunc("/api/gateway/snapshot/import", server.HandleSnapshotImport)

	port := ":8080"
	log.Printf("[EditMind] Enhanced Gateway microservice running on http://localhost%s\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
