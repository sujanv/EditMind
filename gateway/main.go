// EditMind High-Performance Knowledge Gateway & Audit Service
// Intercepts queries, serves fast-path cached knowledge edits, and audits modifications.

package main

import (
	"encoding/json"
	"fmt"
	"log"
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
	Timestamp time.Time `json:"timestamp"`
	LatencyMs float64   `json:"latency_ms"`
}

type QueryRequest struct {
	Prompt string `json:"prompt"`
}

type QueryResponse struct {
	Prompt       string `json:"prompt"`
	Predicted    string `json:"predicted"`
	Intercepted  bool   `json:"intercepted"`
	MatchedEdit  string `json:"matched_edit,omitempty"`
	ResponseTime string `json:"response_time"`
}

type GatewayServer struct {
	mu        sync.RWMutex
	edits     map[string]KnowledgeEdit
	auditLogs []KnowledgeEdit
}

func NewGatewayServer() *GatewayServer {
	return &GatewayServer{
		edits:     make(map[string]KnowledgeEdit),
		auditLogs: make([]KnowledgeEdit, 0),
	}
}

func (s *GatewayServer) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "ok",
		"service":     "EditMind Go Gateway",
		"version":     "0.1.0",
		"total_edits": len(s.edits),
	})
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

	start := time.Now()
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	lowerPrompt := strings.ToLower(req.Prompt)
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Check if prompt matches any active knowledge edit
	intercepted := false
	predicted := "unknown"
	matchedKey := ""

	for subj, edit := range s.edits {
		if strings.Contains(lowerPrompt, subj) {
			intercepted = true
			predicted = edit.Target
			matchedKey = fmt.Sprintf("%s (%s -> %s)", edit.Subject, edit.Editor, edit.Target)
			break
		}
	}

	resp := QueryResponse{
		Prompt:       req.Prompt,
		Predicted:    predicted,
		Intercepted:  intercepted,
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
	http.HandleFunc("/api/gateway/edit", server.HandleRegisterEdit)
	http.HandleFunc("/api/gateway/query", server.HandleQuery)
	http.HandleFunc("/api/gateway/audit", server.HandleAudit)

	port := ":8080"
	log.Printf("[EditMind] Gateway microservice running on http://localhost%s\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
