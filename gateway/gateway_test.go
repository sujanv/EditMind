package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
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

func TestBloomFilterFastNegativeBypass(t *testing.T) {
	bf := NewBloomFilter(4096, 4)
	bf.Add("eiffel")
	bf.Add("tower")
	bf.Add("paris")

	if !bf.MayContain("eiffel") {
		t.Fatalf("Expected bloom filter to contain 'eiffel'")
	}
	if !bf.MayContain("tower") {
		t.Fatalf("Expected bloom filter to contain 'tower'")
	}
	if bf.MayContain("randomnonexistentkeyword12345") {
		t.Fatalf("Expected bloom filter to return false for non-existent keyword")
	}

	if !bf.MayContainAnyWord("The Eiffel Tower is beautiful") {
		t.Fatalf("Expected MayContainAnyWord to find match for prompt")
	}
	if bf.MayContainAnyWord("quantum mechanics in deep space") {
		t.Fatalf("Expected MayContainAnyWord to return false for unrelated query")
	}

	// Test Gateway Query Bloom Bypass
	server := NewGatewayServer()
	edit := KnowledgeEdit{
		Subject: "Eiffel Tower",
		Target:  "Rome",
		Editor:  "rome",
	}
	body, _ := json.Marshal(edit)
	server.HandleRegisterEdit(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/gateway/edit", bytes.NewBuffer(body)))

	// Unrelated query should be bypassed by bloom filter
	unrelatedReq := QueryRequest{Prompt: "quantum mechanics in deep space"}
	unrelatedBody, _ := json.Marshal(unrelatedReq)
	rec := httptest.NewRecorder()
	server.HandleQuery(rec, httptest.NewRequest(http.MethodPost, "/api/gateway/query", bytes.NewBuffer(unrelatedBody)))

	var qResp QueryResponse
	json.NewDecoder(rec.Body).Decode(&qResp)
	if qResp.Intercepted {
		t.Fatalf("Unrelated query should not be intercepted")
	}

	server.mu.RLock()
	bypassCount := server.bloomBypassCount
	server.mu.RUnlock()
	if bypassCount != 1 {
		t.Fatalf("Expected bloom bypass count to be 1, got %d", bypassCount)
	}
}

func TestClusterPeerJoinAndSync(t *testing.T) {
	server := NewGatewayServer()

	// 1. Join peer
	peer := ClusterPeer{
		NodeID:  "node-peer-1",
		Address: "http://10.0.0.2:8080",
	}
	pBody, _ := json.Marshal(peer)
	rec := httptest.NewRecorder()
	server.HandleClusterJoin(rec, httptest.NewRequest(http.MethodPost, "/api/gateway/cluster/join", bytes.NewBuffer(pBody)))
	if rec.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK on cluster join, got %d", rec.Code)
	}

	// 2. Query peers
	recPeers := httptest.NewRecorder()
	server.HandleClusterPeers(recPeers, httptest.NewRequest(http.MethodGet, "/api/gateway/cluster/peers", nil))
	var peersResp struct {
		PeersCount int           `json:"peers_count"`
		Peers      []ClusterPeer `json:"peers"`
	}
	json.NewDecoder(recPeers.Body).Decode(&peersResp)
	if peersResp.PeersCount != 1 {
		t.Fatalf("Expected 1 peer, got %d", peersResp.PeersCount)
	}

	// 3. Cluster sync an edit
	syncEdit := KnowledgeEdit{
		Subject: "Colosseum",
		Target:  "Paris",
		Editor:  "rome",
	}
	syncPayload := map[string]interface{}{
		"sender_id": "node-peer-1",
		"edit":      syncEdit,
	}
	sBody, _ := json.Marshal(syncPayload)
	recSync := httptest.NewRecorder()
	server.HandleClusterSync(recSync, httptest.NewRequest(http.MethodPost, "/api/gateway/cluster/sync", bytes.NewBuffer(sBody)))
	if recSync.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK on cluster sync, got %d", recSync.Code)
	}

	server.mu.RLock()
	_, exists := server.edits["colosseum"]
	server.mu.RUnlock()
	if !exists {
		t.Fatalf("Expected synced edit 'colosseum' to exist in gateway edits")
	}
}

func TestClusterHeartbeatAndPruning(t *testing.T) {
	server := NewGatewayServer()

	// 1. Join peer
	peer := ClusterPeer{
		NodeID:  "node-worker-9",
		Address: "http://10.0.0.9:8080",
	}
	pBody, _ := json.Marshal(peer)
	rec := httptest.NewRecorder()
	server.HandleClusterJoin(rec, httptest.NewRequest(http.MethodPost, "/api/gateway/cluster/join", bytes.NewBuffer(pBody)))
	if rec.Code != http.StatusOK {
		t.Fatalf("Expected 200 on join, got %d", rec.Code)
	}

	// 2. Send heartbeat
	hbPayload := map[string]interface{}{
		"node_id": "node-worker-9",
		"term":    2,
		"role":    "follower",
	}
	hbBody, _ := json.Marshal(hbPayload)
	recHb := httptest.NewRecorder()
	server.HandleClusterHeartbeat(recHb, httptest.NewRequest(http.MethodPost, "/api/gateway/cluster/heartbeat", bytes.NewBuffer(hbBody)))
	if recHb.Code != http.StatusOK {
		t.Fatalf("Expected 200 on heartbeat, got %d", recHb.Code)
	}

	// 3. Test Pruning with small timeout
	server.mu.Lock()
	server.peers["node-worker-9"].LastSeen = time.Now().Add(-10 * time.Minute)
	server.mu.Unlock()

	pruned := server.PruneInactivePeers(5 * time.Minute)
	if pruned != 1 {
		t.Fatalf("Expected 1 pruned peer, got %d", pruned)
	}

	server.mu.RLock()
	st := server.peers["node-worker-9"].Status
	server.mu.RUnlock()
	if st != "offline" {
		t.Fatalf("Expected status offline, got %s", st)
	}
}

func TestClusterLeaderElectionAndState(t *testing.T) {
	server := NewGatewayServer()

	// 1. Trigger election for Candidate Node-A
	electPayload := map[string]interface{}{
		"candidate_id": "node-alpha",
		"term":         5,
	}
	eBody, _ := json.Marshal(electPayload)
	recElect := httptest.NewRecorder()
	server.HandleClusterElect(recElect, httptest.NewRequest(http.MethodPost, "/api/gateway/cluster/elect", bytes.NewBuffer(eBody)))
	if recElect.Code != http.StatusOK {
		t.Fatalf("Expected 200 on elect, got %d", recElect.Code)
	}

	var electResp struct {
		VoteGranted bool   `json:"vote_granted"`
		CurrentTerm uint64 `json:"current_term"`
		LeaderID    string `json:"leader_id"`
	}
	json.NewDecoder(recElect.Body).Decode(&electResp)
	if !electResp.VoteGranted {
		t.Fatalf("Expected vote granted to candidate")
	}
	if electResp.LeaderID != "node-alpha" {
		t.Fatalf("Expected leader to be node-alpha, got %s", electResp.LeaderID)
	}

	// 2. Query cluster state
	recState := httptest.NewRecorder()
	server.HandleClusterState(recState, httptest.NewRequest(http.MethodGet, "/api/gateway/cluster/state", nil))
	if recState.Code != http.StatusOK {
		t.Fatalf("Expected 200 on cluster state, got %d", recState.Code)
	}

	var stateResp struct {
		NodeID         string `json:"node_id"`
		LeaderID       string `json:"leader_id"`
		Term           uint64 `json:"term"`
		WALCommitIndex int    `json:"wal_commit_index"`
	}
	json.NewDecoder(recState.Body).Decode(&stateResp)
	if stateResp.LeaderID != "node-alpha" {
		t.Fatalf("Expected leader_id node-alpha, got %s", stateResp.LeaderID)
	}
	if stateResp.Term != 5 {
		t.Fatalf("Expected term 5, got %d", stateResp.Term)
	}
}

func TestPriorityQueuing(t *testing.T) {
	server := NewGatewayServer()

	// Register an edit
	edit := KnowledgeEdit{
		Subject:   "CEO of Apple",
		Relation:  "is",
		Target:    "Tim Cook",
		OldTarget: "Steve Jobs",
		Editor:    "rome",
	}
	body, _ := json.Marshal(edit)
	server.HandleRegisterEdit(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/gateway/edit", bytes.NewBuffer(body)))

	// 1. High-priority query
	highReq := PriorityQueryRequest{
		Priority: "high",
		Prompt:   "The CEO of Apple is",
	}
	hBody, _ := json.Marshal(highReq)
	recHigh := httptest.NewRecorder()
	server.HandlePriorityQuery(recHigh, httptest.NewRequest(http.MethodPost, "/api/gateway/priority/query", bytes.NewBuffer(hBody)))
	if recHigh.Code != http.StatusOK {
		t.Fatalf("Expected 200 for high priority query, got %d", recHigh.Code)
	}

	var hResp PriorityQueryResponse
	json.NewDecoder(recHigh.Body).Decode(&hResp)
	if !hResp.Intercepted || hResp.Predicted != "Tim Cook" {
		t.Fatalf("Expected Tim Cook, got %s (intercepted: %v)", hResp.Predicted, hResp.Intercepted)
	}
	if hResp.Priority != "high" {
		t.Fatalf("Expected priority 'high', got '%s'", hResp.Priority)
	}

	// 2. Standard-priority query
	stdReq := PriorityQueryRequest{
		Priority: "standard",
		Prompt:   "Who is CEO of Apple?",
	}
	sBody, _ := json.Marshal(stdReq)
	recStd := httptest.NewRecorder()
	server.HandlePriorityQuery(recStd, httptest.NewRequest(http.MethodPost, "/api/gateway/priority/query", bytes.NewBuffer(sBody)))
	if recStd.Code != http.StatusOK {
		t.Fatalf("Expected 200 for standard priority query, got %d", recStd.Code)
	}

	var sResp PriorityQueryResponse
	json.NewDecoder(recStd.Body).Decode(&sResp)
	if sResp.Priority != "standard" {
		t.Fatalf("Expected priority 'standard', got '%s'", sResp.Priority)
	}

	// 3. Verify metrics
	recMetrics := httptest.NewRecorder()
	server.HandleMetrics(recMetrics, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	metricsText := recMetrics.Body.String()
	if !bytes.Contains([]byte(metricsText), []byte("editmind_gateway_high_priority_total 1")) {
		t.Fatalf("Expected high priority metric 1 in output: %s", metricsText)
	}
	if !bytes.Contains([]byte(metricsText), []byte("editmind_gateway_standard_priority_total 1")) {
		t.Fatalf("Expected standard priority metric 1 in output: %s", metricsText)
	}
}

func TestCircuitBreaker(t *testing.T) {
	server := NewGatewayServer()

	// 1. Initial status should be CLOSED
	recStatus := httptest.NewRecorder()
	server.HandleCircuitStatus(recStatus, httptest.NewRequest(http.MethodGet, "/api/gateway/circuit/status", nil))
	var stResp map[string]interface{}
	json.NewDecoder(recStatus.Body).Decode(&stResp)
	if stResp["state"] != "CLOSED" {
		t.Fatalf("Expected CLOSED state initially, got %v", stResp["state"])
	}

	// 2. Trip the circuit breaker
	recTrip := httptest.NewRecorder()
	server.HandleCircuitTrip(recTrip, httptest.NewRequest(http.MethodPost, "/api/gateway/circuit/trip", nil))
	if recTrip.Code != http.StatusOK {
		t.Fatalf("Expected 200 for trip, got %d", recTrip.Code)
	}
	if server.circuitBreaker.State() != StateOpen {
		t.Fatalf("Expected StateOpen after trip")
	}

	// 3. Queries should now be shed with 503
	qReq := PriorityQueryRequest{
		Priority: "standard",
		Prompt:   "Hello world",
	}
	qBody, _ := json.Marshal(qReq)
	recShed := httptest.NewRecorder()
	server.HandlePriorityQuery(recShed, httptest.NewRequest(http.MethodPost, "/api/gateway/priority/query", bytes.NewBuffer(qBody)))
	if recShed.Code != http.StatusServiceUnavailable {
		t.Fatalf("Expected 503 Service Unavailable when circuit open, got %d", recShed.Code)
	}

	// 4. Reset circuit breaker
	recReset := httptest.NewRecorder()
	server.HandleCircuitReset(recReset, httptest.NewRequest(http.MethodPost, "/api/gateway/circuit/reset", nil))
	if recReset.Code != http.StatusOK {
		t.Fatalf("Expected 200 on reset, got %d", recReset.Code)
	}
	if server.circuitBreaker.State() != StateClosed {
		t.Fatalf("Expected StateClosed after reset")
	}

	// 5. Query succeeds after reset
	recOk := httptest.NewRecorder()
	server.HandlePriorityQuery(recOk, httptest.NewRequest(http.MethodPost, "/api/gateway/priority/query", bytes.NewBuffer(qBody)))
	if recOk.Code != http.StatusOK {
		t.Fatalf("Expected 200 after reset, got %d", recOk.Code)
	}
}

func TestSlidingWindowRateLimiter(t *testing.T) {
	limiter := NewSlidingWindowLimiter(100*time.Millisecond, 2)
	if !limiter.Allow() {
		t.Fatalf("1st request should be allowed")
	}
	if !limiter.Allow() {
		t.Fatalf("2nd request should be allowed")
	}
	if limiter.Allow() {
		t.Fatalf("3rd request within window should be rejected")
	}
	if limiter.Count() != 2 {
		t.Fatalf("Expected count 2, got %d", limiter.Count())
	}

	time.Sleep(110 * time.Millisecond)
	if !limiter.Allow() {
		t.Fatalf("Request after window expiration should be allowed")
	}
}

func TestClientReputation(t *testing.T) {
	server := NewGatewayServer()

	// 1. Normal client request
	qReq := QueryRequest{Prompt: "The capital of France is"}
	qBody, _ := json.Marshal(qReq)
	req := httptest.NewRequest(http.MethodPost, "/api/gateway/query", bytes.NewBuffer(qBody))
	req.Header.Set("X-Client-ID", "client-alpha")
	w := httptest.NewRecorder()
	server.HandleQuery(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200 for normal client, got %d", w.Code)
	}

	// 2. Penalize client reputation into quarantine
	for i := 0; i < 4; i++ {
		server.clientRep.RecordViolation("client-alpha")
	}

	// 3. Quarantined client should be blocked with 403
	req2 := httptest.NewRequest(http.MethodPost, "/api/gateway/query", bytes.NewBuffer(qBody))
	req2.Header.Set("X-Client-ID", "client-alpha")
	w2 := httptest.NewRecorder()
	server.HandleQuery(w2, req2)
	if w2.Code != http.StatusForbidden {
		t.Fatalf("Expected 403 Forbidden for quarantined client, got %d", w2.Code)
	}

	// 4. Inspect client stats endpoint
	recStats := httptest.NewRecorder()
	server.HandleClientStats(recStats, httptest.NewRequest(http.MethodGet, "/api/gateway/clients/stats", nil))
	if recStats.Code != http.StatusOK {
		t.Fatalf("Expected 200 on client stats, got %d", recStats.Code)
	}

	// 5. Reset client reputation
	resetBody, _ := json.Marshal(map[string]string{"client_id": "client-alpha"})
	recReset := httptest.NewRecorder()
	server.HandleClientReputationReset(recReset, httptest.NewRequest(http.MethodPost, "/api/gateway/clients/reset", bytes.NewBuffer(resetBody)))
	if recReset.Code != http.StatusOK {
		t.Fatalf("Expected 200 on client reset, got %d", recReset.Code)
	}

	// 6. Request succeeds again
	req3 := httptest.NewRequest(http.MethodPost, "/api/gateway/query", bytes.NewBuffer(qBody))
	req3.Header.Set("X-Client-ID", "client-alpha")
	w3 := httptest.NewRecorder()
	server.HandleQuery(w3, req3)
	if w3.Code != http.StatusOK {
		t.Fatalf("Expected 200 after reset, got %d", w3.Code)
	}
}

