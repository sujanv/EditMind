package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestDistributedMeshQuotaSharing(t *testing.T) {
	rebalancer := NewDistributedMeshRebalancer("node-1", 10*time.Minute)

	// Register 3 peers with varying capacity
	rebalancer.RegisterOrUpdatePeer("peer-a", 100, 100) // 100 available, high surplus
	rebalancer.RegisterOrUpdatePeer("peer-b", 10, 100)  // low capacity
	rebalancer.RegisterOrUpdatePeer("peer-c", 80, 100)  // medium-high capacity

	// peer-b requests 40 tokens
	lease, err := rebalancer.RequestQuotaLease("peer-b", 40)
	if err != nil {
		t.Fatalf("expected lease grant, got err: %v", err)
	}

	if lease.LenderPeerID != "peer-a" {
		t.Errorf("expected lender to be peer-a (highest surplus), got %s", lease.LenderPeerID)
	}
	if lease.TokensGranted != 40 {
		t.Errorf("expected 40 tokens granted, got %d", lease.TokensGranted)
	}

	// Verify telemetry
	telem := rebalancer.GetTelemetry()
	if telem.ActiveLeaseCount != 1 {
		t.Errorf("expected 1 active lease, got %d", telem.ActiveLeaseCount)
	}
	if telem.TotalLentTokens != 40 {
		t.Errorf("expected 40 total lent tokens, got %d", telem.TotalLentTokens)
	}

	// Return lease
	err = rebalancer.ReturnQuotaLease(lease.LeaseID)
	if err != nil {
		t.Fatalf("expected successful lease return, got: %v", err)
	}

	telemAfter := rebalancer.GetTelemetry()
	if telemAfter.ActiveLeaseCount != 0 {
		t.Errorf("expected 0 active leases after return, got %d", telemAfter.ActiveLeaseCount)
	}
	if telemAfter.TotalLentTokens != 0 {
		t.Errorf("expected 0 lent tokens after return, got %d", telemAfter.TotalLentTokens)
	}
}

func TestMeshRebalancerHTTPHandlers(t *testing.T) {
	server := NewGatewayServer()
	server.meshRebalancer.RegisterOrUpdatePeer("lender-node", 200, 200)

	// 1. Request lease via HTTP
	leaseReq := MeshLeaseRequest{
		BorrowerID:      "hungry-node",
		RequestedTokens: 50,
	}
	reqBody, _ := json.Marshal(leaseReq)
	req := httptest.NewRequest(http.MethodPost, "/api/gateway/mesh/lease", bytes.NewReader(reqBody))
	w := httptest.NewRecorder()
	server.handleMeshLease(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var lease TokenLease
	if err := json.NewDecoder(w.Body).Decode(&lease); err != nil {
		t.Fatalf("failed to decode lease JSON: %v", err)
	}
	if lease.TokensGranted != 50 {
		t.Errorf("expected 50 tokens granted, got %d", lease.TokensGranted)
	}

	// 2. Fetch Telemetry
	reqTelem := httptest.NewRequest(http.MethodGet, "/api/gateway/mesh/telemetry", nil)
	wTelem := httptest.NewRecorder()
	server.handleMeshTelemetry(wTelem, reqTelem)

	if wTelem.Code != http.StatusOK {
		t.Fatalf("expected telemetry status 200, got %d", wTelem.Code)
	}

	// 3. Return Lease via HTTP
	returnReq := MeshReturnRequest{LeaseID: lease.LeaseID}
	retBody, _ := json.Marshal(returnReq)
	reqReturn := httptest.NewRequest(http.MethodPost, "/api/gateway/mesh/return", bytes.NewReader(retBody))
	wReturn := httptest.NewRecorder()
	server.handleMeshReturn(wReturn, reqReturn)

	if wReturn.Code != http.StatusOK {
		t.Fatalf("expected return status 200, got %d", wReturn.Code)
	}
}
