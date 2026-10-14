package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	vecA := []float64{1.0, 0.0, 0.0}
	vecB := []float64{1.0, 0.0, 0.0}
	sim := CosineSimilarity(vecA, vecB)
	if sim < 0.999 {
		t.Fatalf("Expected similarity ~1.0, got %f", sim)
	}

	vecC := []float64{0.0, 1.0, 0.0}
	simOrthogonal := CosineSimilarity(vecA, vecC)
	if simOrthogonal != 0.0 {
		t.Fatalf("Expected similarity 0.0, got %f", simOrthogonal)
	}
}

func TestTokenBucketRateLimiter(t *testing.T) {
	tb := NewTokenBucket(2.0, 1.0)
	if !tb.Allow() {
		t.Fatalf("First request should be allowed")
	}
	if !tb.Allow() {
		t.Fatalf("Second request should be allowed")
	}
	if tb.Allow() {
		t.Fatalf("Third request should be throttled")
	}
}

func TestGatewayRegistrationAndQuery(t *testing.T) {
	server := NewGatewayServer()

	// Register an edit
	edit := KnowledgeEdit{
		Subject:   "Eiffel Tower",
		Relation:  "located in",
		Target:    "Rome",
		OldTarget: "Paris",
		Editor:    "rome",
		Embedding: []float64{0.5, 0.5, 0.0},
	}
	body, _ := json.Marshal(edit)
	req := httptest.NewRequest(http.MethodPost, "/api/gateway/edit", bytes.NewBuffer(body))
	w := httptest.NewRecorder()
	server.HandleRegisterEdit(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("Expected 201 Created, got %d", w.Code)
	}

	// Query with match
	qReq := QueryRequest{Prompt: "The Eiffel Tower is in"}
	qBody, _ := json.Marshal(qReq)
	req2 := httptest.NewRequest(http.MethodPost, "/api/gateway/query", bytes.NewBuffer(qBody))
	w2 := httptest.NewRecorder()
	server.HandleQuery(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", w2.Code)
	}

	var resp QueryResponse
	json.NewDecoder(w2.Body).Decode(&resp)
	if !resp.Intercepted {
		t.Fatalf("Expected query to be intercepted")
	}
	if resp.Predicted != "Rome" {
		t.Fatalf("Expected predicted Rome, got %s", resp.Predicted)
	}
}

func TestMultiHopResolution(t *testing.T) {
	server := NewGatewayServer()

	// Hop 1: Messi -> Inter Miami
	edit1 := KnowledgeEdit{
		Subject:  "Messi",
		Relation: "plays for",
		Target:   "Inter Miami",
	}
	// Hop 2: Inter Miami -> Chase Stadium
	edit2 := KnowledgeEdit{
		Subject:  "Inter Miami",
		Relation: "stadium",
		Target:   "Chase Stadium",
	}

	body1, _ := json.Marshal(edit1)
	req1 := httptest.NewRequest(http.MethodPost, "/api/gateway/edit", bytes.NewBuffer(body1))
	server.HandleRegisterEdit(httptest.NewRecorder(), req1)

	body2, _ := json.Marshal(edit2)
	req2 := httptest.NewRequest(http.MethodPost, "/api/gateway/edit", bytes.NewBuffer(body2))
	server.HandleRegisterEdit(httptest.NewRecorder(), req2)

	// Test multi-hop resolution
	target, path, found := server.ResolveMultiHop("Messi", "stadium")
	if !found {
		t.Fatalf("Expected multi-hop chain to be found")
	}
	if target != "Chase Stadium" {
		t.Fatalf("Expected target 'Chase Stadium', got '%s'", target)
	}
	if len(path) != 3 {
		t.Fatalf("Expected path length 3 [Messi -> Inter Miami -> Chase Stadium], got %d", len(path))
	}

	// Test via HTTP endpoint
	mReq := MultiHopChainRequest{AnchorSubject: "Messi", HopRelation: "stadium"}
	mBody, _ := json.Marshal(mReq)
	httpReq := httptest.NewRequest(http.MethodPost, "/api/gateway/multihop", bytes.NewBuffer(mBody))
	w := httptest.NewRecorder()
	server.HandleMultiHopResolve(w, httpReq)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", w.Code)
	}
	var mResp MultiHopChainResponse
	json.NewDecoder(w.Body).Decode(&mResp)
	if !mResp.Found || mResp.ResolvedTarget != "Chase Stadium" {
		t.Fatalf("Endpoint returned unexpected response: %+v", mResp)
	}
}

func TestWriteAheadLogAndSnapshot(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := tmpDir + "/test_wal.log"

	server1 := NewGatewayServer(walPath)
	edit := KnowledgeEdit{
		Subject:  "Saturn",
		Relation: "has rings",
		Target:   "Yes",
		Editor:   "rome",
	}
	body, _ := json.Marshal(edit)
	req := httptest.NewRequest(http.MethodPost, "/api/gateway/edit", bytes.NewBuffer(body))
	server1.HandleRegisterEdit(httptest.NewRecorder(), req)

	// Verify WAL file was created and contains edit
	server2 := NewGatewayServer(walPath)
	server2.mu.RLock()
	replayedEdit, exists := server2.edits["saturn"]
	server2.mu.RUnlock()
	if !exists {
		t.Fatalf("Expected 'saturn' to be recovered from WAL")
	}
	if replayedEdit.Target != "Yes" {
		t.Fatalf("Expected target 'Yes', got '%s'", replayedEdit.Target)
	}

	// Test snapshot export
	expReq := httptest.NewRequest(http.MethodGet, "/api/gateway/snapshot/export", nil)
	expRec := httptest.NewRecorder()
	server2.HandleSnapshotExport(expRec, expReq)
	if expRec.Code != http.StatusOK {
		t.Fatalf("Snapshot export failed: %d", expRec.Code)
	}

	// Test snapshot import into fresh server
	server3 := NewGatewayServer()
	impReq := httptest.NewRequest(http.MethodPost, "/api/gateway/snapshot/import", expRec.Body)
	impRec := httptest.NewRecorder()
	server3.HandleSnapshotImport(impRec, impReq)
	if impRec.Code != http.StatusOK {
		t.Fatalf("Snapshot import failed: %d", impRec.Code)
	}

	server3.mu.RLock()
	_, exists3 := server3.edits["saturn"]
	server3.mu.RUnlock()
	if !exists3 {
		t.Fatalf("Expected 'saturn' to exist after snapshot import")
	}
}

func TestLatencyPercentiles(t *testing.T) {
	server := NewGatewayServer()
	server.mu.Lock()
	server.latencies = []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
	server.mu.Unlock()

	p50, p90, p99 := server.CalculatePercentiles()
	if p50 < 5.0 || p50 > 6.0 {
		t.Fatalf("Expected p50 ~5.5, got %f", p50)
	}
	if p90 < 9.0 {
		t.Fatalf("Expected p90 >= 9.0, got %f", p90)
	}
	if p99 < 9.0 {
		t.Fatalf("Expected p99 >= 9.0, got %f", p99)
	}
}
