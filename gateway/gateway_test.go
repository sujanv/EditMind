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
