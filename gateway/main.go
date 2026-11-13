// EditMind High-Performance Knowledge Gateway & Audit Service
// Features: Vector Similarity Cache, Token-Bucket Rate Limiter, Bloom Filter Fast-Negative Bypass,
// Write-Ahead Log (WAL), Latency Percentiles, Distributed Peer Replication, & Prometheus Metrics.

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

// BloomFilter provides sub-microsecond negative existence checks on edited facts.
type BloomFilter struct {
	bits []uint64
	size uint64
	k    int
	mu   sync.RWMutex
}

func NewBloomFilter(sizeBits uint64, k int) *BloomFilter {
	if sizeBits == 0 {
		sizeBits = 8192
	}
	if k <= 0 {
		k = 4
	}
	words := (sizeBits + 63) / 64
	return &BloomFilter{
		bits: make([]uint64, words),
		size: words * 64,
		k:    k,
	}
}

func (bf *BloomFilter) hash(s string, seed int) uint64 {
	var h uint64 = 14695981039346656037 ^ (uint64(seed) * 1099511628211)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h % bf.size
}

func (bf *BloomFilter) Add(item string) {
	bf.mu.Lock()
	defer bf.mu.Unlock()
	clean := strings.ToLower(strings.TrimSpace(item))
	if clean == "" {
		return
	}
	for i := 0; i < bf.k; i++ {
		idx := bf.hash(clean, i)
		bf.bits[idx/64] |= (1 << (idx % 64))
	}
}

func (bf *BloomFilter) MayContain(item string) bool {
	bf.mu.RLock()
	defer bf.mu.RUnlock()
	clean := strings.ToLower(strings.TrimSpace(item))
	if clean == "" {
		return false
	}
	for i := 0; i < bf.k; i++ {
		idx := bf.hash(clean, i)
		if (bf.bits[idx/64] & (1 << (idx % 64))) == 0 {
			return false
		}
	}
	return true
}

func (bf *BloomFilter) MayContainAnyWord(prompt string) bool {
	words := strings.Fields(strings.ToLower(prompt))
	for _, w := range words {
		clean := strings.Trim(w, ".,;:?!'\"()[]{}")
		if len(clean) >= 2 && bf.MayContain(clean) {
			return true
		}
	}
	return false
}

type CircuitState string

const (
	StateClosed   CircuitState = "CLOSED"
	StateHalfOpen CircuitState = "HALF_OPEN"
	StateOpen     CircuitState = "OPEN"
)

type CircuitBreaker struct {
	mu               sync.Mutex
	state            CircuitState
	failureThreshold int
	recoveryTimeout  time.Duration
	failureCount     int
	successCount     int
	halfOpenSuccess  int
	lastStateChange  time.Time
}

func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
	if threshold <= 0 {
		threshold = 5
	}
	if timeout <= 0 {
		timeout = 500 * time.Millisecond
	}
	return &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: threshold,
		recoveryTimeout:  timeout,
		lastStateChange:  time.Now(),
	}
}

func (cb *CircuitBreaker) Allow() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()
	if cb.state == StateOpen {
		if now.Sub(cb.lastStateChange) >= cb.recoveryTimeout {
			cb.state = StateHalfOpen
			cb.lastStateChange = now
			cb.halfOpenSuccess = 0
			return true
		}
		return false
	}
	return true
}

func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.successCount++
	if cb.state == StateHalfOpen {
		cb.halfOpenSuccess++
		if cb.halfOpenSuccess >= 2 {
			cb.state = StateClosed
			cb.failureCount = 0
			cb.lastStateChange = time.Now()
		}
	} else if cb.state == StateClosed {
		cb.failureCount = 0
	}
}

func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount++
	if cb.state == StateHalfOpen || cb.failureCount >= cb.failureThreshold {
		cb.state = StateOpen
		cb.lastStateChange = time.Now()
	}
}

func (cb *CircuitBreaker) State() CircuitState {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.state
}

func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.state = StateClosed
	cb.failureCount = 0
	cb.successCount = 0
	cb.halfOpenSuccess = 0
	cb.lastStateChange = time.Now()
}

func (cb *CircuitBreaker) Trip() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.state = StateOpen
	cb.lastStateChange = time.Now()
}

type PriorityQueryRequest struct {
	ID        string    `json:"id,omitempty"`
	Priority  string    `json:"priority"` // "high" or "standard"
	Prompt    string    `json:"prompt"`
	Embedding []float64 `json:"embedding,omitempty"`
}

type PriorityQueryResponse struct {
	QueryResponse
	Priority     string  `json:"priority"`
	QueueWaitMs  float64 `json:"queue_wait_ms"`
	CircuitState string  `json:"circuit_state"`
}

type SlidingWindowLimiter struct {
	mu          sync.Mutex
	window      time.Duration
	maxRequests int
	timestamps  []time.Time
}

func NewSlidingWindowLimiter(window time.Duration, maxRequests int) *SlidingWindowLimiter {
	return &SlidingWindowLimiter{
		window:      window,
		maxRequests: maxRequests,
		timestamps:  make([]time.Time, 0, maxRequests*2),
	}
}

func (sw *SlidingWindowLimiter) Allow() bool {
	sw.mu.Lock()
	defer sw.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-sw.window)

	i := 0
	for i < len(sw.timestamps) && sw.timestamps[i].Before(cutoff) {
		i++
	}
	if i > 0 {
		sw.timestamps = sw.timestamps[i:]
	}

	if len(sw.timestamps) < sw.maxRequests {
		sw.timestamps = append(sw.timestamps, now)
		return true
	}
	return false
}

func (sw *SlidingWindowLimiter) Count() int {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	cutoff := time.Now().Add(-sw.window)
	count := 0
	for _, t := range sw.timestamps {
		if t.After(cutoff) {
			count++
		}
	}
	return count
}

type ClientReputation struct {
	ClientID      string    `json:"client_id"`
	Score         float64   `json:"score"`
	TotalRequests int64     `json:"total_requests"`
	Violations    int       `json:"violations"`
	Status        string    `json:"status"` // "active", "degraded", "quarantined"
	LastSeen      time.Time `json:"last_seen"`
}

type ClientReputationManager struct {
	mu      sync.RWMutex
	clients map[string]*ClientReputation
}

func NewClientReputationManager() *ClientReputationManager {
	return &ClientReputationManager{
		clients: make(map[string]*ClientReputation),
	}
}

func (crm *ClientReputationManager) RecordRequest(clientID string) (bool, string) {
	if clientID == "" {
		return true, "active"
	}
	crm.mu.Lock()
	defer crm.mu.Unlock()

	client, exists := crm.clients[clientID]
	if !exists {
		client = &ClientReputation{
			ClientID:      clientID,
			Score:         100.0,
			TotalRequests: 1,
			Violations:    0,
			Status:        "active",
			LastSeen:      time.Now(),
		}
		crm.clients[clientID] = client
		return true, "active"
	}

	client.TotalRequests++
	client.LastSeen = time.Now()
	if client.Score < 100.0 {
		client.Score = math.Min(100.0, client.Score+0.2)
	}

	if client.Status == "quarantined" {
		return false, "quarantined"
	}
	return true, client.Status
}

func (crm *ClientReputationManager) RecordViolation(clientID string) {
	if clientID == "" {
		return
	}
	crm.mu.Lock()
	defer crm.mu.Unlock()

	client, exists := crm.clients[clientID]
	if !exists {
		client = &ClientReputation{
			ClientID: clientID,
			Score:    100.0,
			LastSeen: time.Now(),
		}
		crm.clients[clientID] = client
	}

	client.Violations++
	client.Score = math.Max(0.0, client.Score-20.0)
	if client.Score < 40.0 {
		client.Status = "quarantined"
	} else if client.Score < 70.0 {
		client.Status = "degraded"
	}
}

func (crm *ClientReputationManager) Reset(clientID string) {
	crm.mu.Lock()
	defer crm.mu.Unlock()
	if client, exists := crm.clients[clientID]; exists {
		client.Score = 100.0
		client.Status = "active"
		client.Violations = 0
	}
}

func (crm *ClientReputationManager) GetAll() []ClientReputation {
	crm.mu.RLock()
	defer crm.mu.RUnlock()
	res := make([]ClientReputation, 0, len(crm.clients))
	for _, c := range crm.clients {
		res = append(res, *c)
	}
	return res
}

type ClusterPeer struct {
	NodeID   string    `json:"node_id"`
	Address  string    `json:"address"`
	JoinedAt time.Time `json:"joined_at"`
	LastSeen time.Time `json:"last_seen"`
	Status   string    `json:"status"` // "active", "offline"
	Role     string    `json:"role"`   // "leader", "follower"
}

type GatewayServer struct {
	mu             sync.RWMutex
	nodeID         string
	leaderID       string
	currentTerm    uint64
	isLeader       bool
	edits          map[string]KnowledgeEdit
	auditLogs      []KnowledgeEdit
	rateLimiter    *TokenBucket
	circuitBreaker *CircuitBreaker
	slidingLimiter *SlidingWindowLimiter
	clientRep      *ClientReputationManager
	walPath        string
	walMu          sync.Mutex
	bloom          *BloomFilter
	peers          map[string]*ClusterPeer

	// Metrics and latency counters
	totalRequests         int64
	interceptedCount      int64
	rateLimitedCount      int64
	bloomBypassCount      int64
	highPriorityCount     int64
	standardPriorityCount int64
	latencies             []float64
}

func NewGatewayServer(walPath ...string) *GatewayServer {
	path := ""
	if len(walPath) > 0 {
		path = walPath[0]
	}
	s := &GatewayServer{
		nodeID:         fmt.Sprintf("node-%d", time.Now().UnixNano()%10000),
		leaderID:       "",
		currentTerm:    1,
		isLeader:       false,
		edits:          make(map[string]KnowledgeEdit),
		auditLogs:      make([]KnowledgeEdit, 0),
		rateLimiter:    NewTokenBucket(100.0, 50.0), // 100 capacity, 50 req/sec refill
		circuitBreaker: NewCircuitBreaker(5, 500*time.Millisecond),
		slidingLimiter: NewSlidingWindowLimiter(10*time.Second, 500),
		clientRep:      NewClientReputationManager(),
		walPath:        path,
		bloom:          NewBloomFilter(16384, 4),
		peers:          make(map[string]*ClusterPeer),
		latencies:      make([]float64, 0, 1000),
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
		"version":     "0.3.0",
		"node_id":     s.nodeID,
		"total_edits": len(s.edits),
		"peers_count": len(s.peers),
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

	fmt.Fprintf(w, "# HELP editmind_gateway_bloom_bypasses Total negative queries bypassed instantly by Bloom Filter.\n")
	fmt.Fprintf(w, "# TYPE editmind_gateway_bloom_bypasses counter\n")
	fmt.Fprintf(w, "editmind_gateway_bloom_bypasses %d\n", s.bloomBypassCount)

	fmt.Fprintf(w, "# HELP editmind_gateway_high_priority_total High-priority requests processed.\n")
	fmt.Fprintf(w, "# TYPE editmind_gateway_high_priority_total counter\n")
	fmt.Fprintf(w, "editmind_gateway_high_priority_total %d\n", s.highPriorityCount)

	fmt.Fprintf(w, "# HELP editmind_gateway_standard_priority_total Standard-priority requests processed.\n")
	fmt.Fprintf(w, "# TYPE editmind_gateway_standard_priority_total counter\n")
	fmt.Fprintf(w, "editmind_gateway_standard_priority_total %d\n", s.standardPriorityCount)

	fmt.Fprintf(w, "# HELP editmind_gateway_circuit_state Current circuit breaker state (0=closed, 1=half-open, 2=open).\n")
	fmt.Fprintf(w, "# TYPE editmind_gateway_circuit_state gauge\n")
	cbVal := 0
	if s.circuitBreaker.State() == StateHalfOpen {
		cbVal = 1
	} else if s.circuitBreaker.State() == StateOpen {
		cbVal = 2
	}
	fmt.Fprintf(w, "editmind_gateway_circuit_state %d\n", cbVal)

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

	if edit.Timestamp.IsZero() {
		edit.Timestamp = time.Now()
	}
	key := strings.ToLower(strings.TrimSpace(edit.Subject))

	s.mu.Lock()
	s.edits[key] = edit
	s.auditLogs = append(s.auditLogs, edit)
	s.mu.Unlock()

	// Update bloom filter with subject and its words
	s.bloom.Add(key)
	for _, word := range strings.Fields(key) {
		s.bloom.Add(word)
	}

	_ = s.AppendWAL(edit)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "registered",
		"edit":   edit,
	})
}

func (s *GatewayServer) executeInternalQuery(prompt string, embedding []float64, start time.Time) QueryResponse {
	s.mu.Lock()
	s.totalRequests++
	s.mu.Unlock()

	lowerPrompt := strings.ToLower(prompt)

	// Bloom Filter Fast-Negative Check:
	if len(embedding) == 0 && !s.bloom.MayContainAnyWord(lowerPrompt) {
		s.mu.Lock()
		s.bloomBypassCount++
		s.mu.Unlock()

		return QueryResponse{
			Prompt:       prompt,
			Predicted:    "unknown",
			Intercepted:  false,
			Similarity:   0.0,
			ResponseTime: time.Since(start).String(),
		}
	}

	s.mu.RLock()
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
	if !intercepted && len(embedding) > 0 {
		for _, edit := range s.edits {
			if len(edit.Embedding) > 0 {
				sim := CosineSimilarity(embedding, edit.Embedding)
				if sim > 0.85 && sim > bestSim {
					bestSim = sim
					intercepted = true
					predicted = edit.Target
					matchedKey = fmt.Sprintf("semantic_match::%s (sim: %.3f)", edit.Subject, sim)
				}
			}
		}
	}
	s.mu.RUnlock()

	if intercepted {
		s.mu.Lock()
		s.interceptedCount++
		s.mu.Unlock()
	}

	elapsedMs := float64(time.Since(start).Microseconds()) / 1000.0
	s.mu.Lock()
	s.latencies = append(s.latencies, elapsedMs)
	if len(s.latencies) > 5000 {
		s.latencies = s.latencies[1000:]
	}
	s.mu.Unlock()

	return QueryResponse{
		Prompt:       prompt,
		Predicted:    predicted,
		Intercepted:  intercepted,
		Similarity:   bestSim,
		MatchedEdit:  matchedKey,
		ResponseTime: time.Since(start).String(),
	}
}

func (s *GatewayServer) HandleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	clientID := r.Header.Get("X-Client-ID")
	if clientID != "" {
		allowed, _ := s.clientRep.RecordRequest(clientID)
		if !allowed {
			http.Error(w, fmt.Sprintf("Client '%s' is quarantined due to low reputation score", clientID), http.StatusForbidden)
			return
		}
	}

	if !s.slidingLimiter.Allow() {
		if clientID != "" {
			s.clientRep.RecordViolation(clientID)
		}
		http.Error(w, "Sliding-window rate limit exceeded", http.StatusTooManyRequests)
		return
	}

	if !s.rateLimiter.Allow() {
		s.mu.Lock()
		s.rateLimitedCount++
		s.mu.Unlock()
		if clientID != "" {
			s.clientRep.RecordViolation(clientID)
		}
		http.Error(w, "Rate limit exceeded (token bucket exhausted)", http.StatusTooManyRequests)
		return
	}

	start := time.Now()
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	resp := s.executeInternalQuery(req.Prompt, req.Embedding, start)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *GatewayServer) HandlePriorityQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	clientID := r.Header.Get("X-Client-ID")
	if clientID != "" {
		allowed, _ := s.clientRep.RecordRequest(clientID)
		if !allowed {
			http.Error(w, fmt.Sprintf("Client '%s' is quarantined due to low reputation score", clientID), http.StatusForbidden)
			return
		}
	}

	if !s.circuitBreaker.Allow() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error":         "Circuit breaker is OPEN - query shed",
			"circuit_state": s.circuitBreaker.State(),
		})
		return
	}

	start := time.Now()
	var req PriorityQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.circuitBreaker.RecordFailure()
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	tier := strings.ToLower(strings.TrimSpace(req.Priority))
	if tier == "" {
		tier = "standard"
	}

	// Track priority metrics
	s.mu.Lock()
	if tier == "high" {
		s.highPriorityCount++
	} else {
		s.standardPriorityCount++
	}
	s.mu.Unlock()

	// High priority bypasses rate limiter; standard checks sliding limiter and token bucket
	if tier != "high" {
		if !s.slidingLimiter.Allow() {
			if clientID != "" {
				s.clientRep.RecordViolation(clientID)
			}
			http.Error(w, "Sliding-window rate limit exceeded", http.StatusTooManyRequests)
			return
		}
		if !s.rateLimiter.Allow() {
			s.mu.Lock()
			s.rateLimitedCount++
			s.mu.Unlock()
			if clientID != "" {
				s.clientRep.RecordViolation(clientID)
			}
			http.Error(w, "Rate limit exceeded for standard tier", http.StatusTooManyRequests)
			return
		}
	}

	baseResp := s.executeInternalQuery(req.Prompt, req.Embedding, start)
	s.circuitBreaker.RecordSuccess()

	pResp := PriorityQueryResponse{
		QueryResponse: baseResp,
		Priority:      tier,
		QueueWaitMs:   0.05,
		CircuitState:  string(s.circuitBreaker.State()),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(pResp)
}

func (s *GatewayServer) HandleCircuitStatus(w http.ResponseWriter, r *http.Request) {
	s.circuitBreaker.mu.Lock()
	state := s.circuitBreaker.state
	failCount := s.circuitBreaker.failureCount
	succCount := s.circuitBreaker.successCount
	timeoutMs := s.circuitBreaker.recoveryTimeout.Milliseconds()
	lastChange := s.circuitBreaker.lastStateChange
	s.circuitBreaker.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"state":               state,
		"failure_count":       failCount,
		"success_count":       succCount,
		"recovery_timeout_ms": timeoutMs,
		"last_state_change":   lastChange,
	})
}

func (s *GatewayServer) HandleCircuitReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.circuitBreaker.Reset()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":        "circuit_reset",
		"circuit_state": s.circuitBreaker.State(),
	})
}

func (s *GatewayServer) HandleCircuitTrip(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.circuitBreaker.Trip()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":        "circuit_tripped",
		"circuit_state": s.circuitBreaker.State(),
	})
}

func (s *GatewayServer) HandleClientStats(w http.ResponseWriter, r *http.Request) {
	clients := s.clientRep.GetAll()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"total_clients": len(clients),
		"clients":       clients,
	})
}

func (s *GatewayServer) HandleClientReputationReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		ClientID string `json:"client_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.clientRep.Reset(req.ClientID)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "reset",
		"client_id": req.ClientID,
	})
}

func (s *GatewayServer) HandleSlidingStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"sliding_window_sec": 10,
		"active_requests":    s.slidingLimiter.Count(),
		"max_requests":       s.slidingLimiter.maxRequests,
	})
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
			s.bloom.Add(key)
			for _, w := range strings.Fields(key) {
				s.bloom.Add(w)
			}
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
		"node_id":            s.nodeID,
		"total_requests":     s.totalRequests,
		"intercepted_count":  s.interceptedCount,
		"rate_limited_count": s.rateLimitedCount,
		"bloom_bypass_count": s.bloomBypassCount,
		"active_edits_count": len(s.edits),
		"latency_p50_ms":     p50,
		"latency_p90_ms":     p90,
		"latency_p99_ms":     p99,
		"cluster_peers":      len(s.peers),
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
		s.bloom.Add(k)
		for _, w := range strings.Fields(k) {
			s.bloom.Add(w)
		}
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

// HandleClusterJoin allows external gateway instances to join the cluster.
func (s *GatewayServer) HandleClusterJoin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var peer ClusterPeer
	if err := json.NewDecoder(r.Body).Decode(&peer); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if peer.NodeID == "" || peer.Address == "" {
		http.Error(w, "Missing node_id or address", http.StatusBadRequest)
		return
	}
	peer.JoinedAt = time.Now()
	peer.LastSeen = time.Now()
	if peer.Status == "" {
		peer.Status = "active"
	}
	if peer.Role == "" {
		peer.Role = "follower"
	}

	s.mu.Lock()
	s.peers[peer.NodeID] = &peer
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "joined",
		"node_id":     s.nodeID,
		"total_peers": len(s.peers),
	})
}

// HandleClusterHeartbeat updates a peer's heartbeat and returns leader and term status.
func (s *GatewayServer) HandleClusterHeartbeat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var hb struct {
		NodeID string `json:"node_id"`
		Term   uint64 `json:"term"`
		Role   string `json:"role"`
	}
	if err := json.NewDecoder(r.Body).Decode(&hb); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	if peer, exists := s.peers[hb.NodeID]; exists {
		peer.LastSeen = time.Now()
		peer.Status = "active"
		if hb.Role != "" {
			peer.Role = hb.Role
		}
	}
	s.mu.Unlock()

	s.mu.RLock()
	defer s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "acknowledged",
		"node_id":   s.nodeID,
		"term":      s.currentTerm,
		"leader_id": s.leaderID,
		"is_leader": s.isLeader,
	})
}

// HandleClusterElect initiates leader election across the cluster.
func (s *GatewayServer) HandleClusterElect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		CandidateID string `json:"candidate_id"`
		Term        uint64 `json:"term"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	granted := false
	if req.Term > s.currentTerm || (req.Term == s.currentTerm && s.leaderID == "") {
		s.currentTerm = req.Term
		s.leaderID = req.CandidateID
		s.isLeader = (req.CandidateID == s.nodeID)
		granted = true
	} else if req.CandidateID == s.nodeID {
		s.currentTerm++
		s.leaderID = s.nodeID
		s.isLeader = true
		granted = true
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"vote_granted": granted,
		"current_term": s.currentTerm,
		"leader_id":    s.leaderID,
		"is_leader":    s.isLeader,
	})
}

// PruneInactivePeers checks and updates status of stale peers.
func (s *GatewayServer) PruneInactivePeers(timeout time.Duration) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	pruned := 0
	for _, p := range s.peers {
		if now.Sub(p.LastSeen) > timeout && p.Status == "active" {
			p.Status = "offline"
			pruned++
		}
	}
	return pruned
}

// HandleClusterState returns full distributed cluster topology, consensus state, and WAL watermark.
func (s *GatewayServer) HandleClusterState(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	activePeers := 0
	for _, p := range s.peers {
		if p.Status == "active" {
			activePeers++
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"node_id":           s.nodeID,
		"is_leader":         s.isLeader,
		"leader_id":         s.leaderID,
		"term":              s.currentTerm,
		"total_edits":       len(s.edits),
		"wal_commit_index":  len(s.auditLogs),
		"active_peers":      activePeers,
		"total_peers":       len(s.peers),
	})
}

// HandleClusterPeers returns all known cluster peers.
func (s *GatewayServer) HandleClusterPeers(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	peerList := make([]ClusterPeer, 0, len(s.peers))
	for _, p := range s.peers {
		peerList = append(peerList, *p)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"node_id":     s.nodeID,
		"peers_count": len(peerList),
		"peers":       peerList,
	})
}

// HandleClusterSync ingests a replicated edit from another node in the cluster.
func (s *GatewayServer) HandleClusterSync(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var syncPayload struct {
		SenderID string        `json:"sender_id"`
		Edit     KnowledgeEdit `json:"edit"`
	}
	if err := json.NewDecoder(r.Body).Decode(&syncPayload); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	key := strings.ToLower(strings.TrimSpace(syncPayload.Edit.Subject))
	s.mu.Lock()
	s.edits[key] = syncPayload.Edit
	s.auditLogs = append(s.auditLogs, syncPayload.Edit)
	s.mu.Unlock()

	s.bloom.Add(key)
	for _, w := range strings.Fields(key) {
		s.bloom.Add(w)
	}
	_ = s.AppendWAL(syncPayload.Edit)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "synchronized",
		"subject":     syncPayload.Edit.Subject,
		"total_edits": len(s.edits),
	})
}

func main() {
	server := NewGatewayServer("gateway_wal.log")

	http.HandleFunc("/health", server.HandleHealth)
	http.HandleFunc("/metrics", server.HandleMetrics)
	http.HandleFunc("/api/gateway/edit", server.HandleRegisterEdit)
	http.HandleFunc("/api/gateway/query", server.HandleQuery)
	http.HandleFunc("/api/gateway/priority/query", server.HandlePriorityQuery)
	http.HandleFunc("/api/gateway/circuit/status", server.HandleCircuitStatus)
	http.HandleFunc("/api/gateway/circuit/reset", server.HandleCircuitReset)
	http.HandleFunc("/api/gateway/circuit/trip", server.HandleCircuitTrip)
	http.HandleFunc("/api/gateway/clients/stats", server.HandleClientStats)
	http.HandleFunc("/api/gateway/clients/reset", server.HandleClientReputationReset)
	http.HandleFunc("/api/gateway/sliding/stats", server.HandleSlidingStats)
	http.HandleFunc("/api/gateway/multihop", server.HandleMultiHopResolve)
	http.HandleFunc("/api/gateway/audit", server.HandleAudit)
	http.HandleFunc("/api/gateway/stats", server.HandleStats)
	http.HandleFunc("/api/gateway/snapshot/export", server.HandleSnapshotExport)
	http.HandleFunc("/api/gateway/snapshot/import", server.HandleSnapshotImport)
	http.HandleFunc("/api/gateway/cluster/join", server.HandleClusterJoin)
	http.HandleFunc("/api/gateway/cluster/peers", server.HandleClusterPeers)
	http.HandleFunc("/api/gateway/cluster/sync", server.HandleClusterSync)
	http.HandleFunc("/api/gateway/cluster/heartbeat", server.HandleClusterHeartbeat)
	http.HandleFunc("/api/gateway/cluster/elect", server.HandleClusterElect)
	http.HandleFunc("/api/gateway/cluster/state", server.HandleClusterState)

	port := ":8080"
	log.Printf("[EditMind] Enhanced Gateway microservice running on http://localhost%s (Node: %s)\n", port, server.nodeID)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
