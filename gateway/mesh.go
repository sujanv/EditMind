package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"
)

// PeerQuota represents an individual gateway replica's token allocation.
type PeerQuota struct {
	PeerID          string    `json:"peer_id"`
	AvailableTokens int       `json:"available_tokens"`
	MaxCapacity     int       `json:"max_capacity"`
	BorrowedTokens  int       `json:"borrowed_tokens"`
	LentTokens      int       `json:"lent_tokens"`
	LastUpdated     time.Time `json:"last_updated"`
}

// TokenLease represents an active dynamic lease between two cluster replicas.
type TokenLease struct {
	LeaseID       string    `json:"lease_id"`
	LenderPeerID  string    `json:"lender_peer_id"`
	BorrowerID    string    `json:"borrower_id"`
	TokensGranted int       `json:"tokens_granted"`
	IssuedAt      time.Time `json:"issued_at"`
	ExpiresAt     time.Time `json:"expires_at"`
}

// MeshTelemetry holds cluster-wide quota distribution metrics.
type MeshTelemetry struct {
	TotalClusterCapacity int       `json:"total_cluster_capacity"`
	TotalAvailableTokens int       `json:"total_available_tokens"`
	TotalLentTokens      int       `json:"total_lent_tokens"`
	ActiveLeaseCount     int       `json:"active_lease_count"`
	ClusterSaturationPct float64   `json:"cluster_saturation_pct"`
	Peers                []PeerQuota `json:"peers"`
}

// DistributedMeshRebalancer coordinates dynamic token quota sharing across cluster peers.
type DistributedMeshRebalancer struct {
	mu           sync.RWMutex
	nodeID       string
	peers        map[string]*PeerQuota
	activeLeases map[string]*TokenLease
	leaseTTL     time.Duration
}

// NewDistributedMeshRebalancer initializes a new cluster quota rebalancer.
func NewDistributedMeshRebalancer(nodeID string, leaseTTL time.Duration) *DistributedMeshRebalancer {
	return &DistributedMeshRebalancer{
		nodeID:       nodeID,
		peers:        make(map[string]*PeerQuota),
		activeLeases: make(map[string]*TokenLease),
		leaseTTL:     leaseTTL,
	}
}

// RegisterOrUpdatePeer registers or updates capacity for a peer node.
func (dmr *DistributedMeshRebalancer) RegisterOrUpdatePeer(peerID string, available int, maxCap int) {
	dmr.mu.Lock()
	defer dmr.mu.Unlock()

	p, exists := dmr.peers[peerID]
	if !exists {
		dmr.peers[peerID] = &PeerQuota{
			PeerID:          peerID,
			AvailableTokens: available,
			MaxCapacity:     maxCap,
			LastUpdated:     time.Now(),
		}
	} else {
		p.AvailableTokens = available
		p.MaxCapacity = maxCap
		p.LastUpdated = time.Now()
	}
}

// RequestQuotaLease transfers tokens from the most idle peer to the requesting peer.
func (dmr *DistributedMeshRebalancer) RequestQuotaLease(borrowerID string, requestedTokens int) (*TokenLease, error) {
	dmr.mu.Lock()
	defer dmr.mu.Unlock()

	dmr.cleanupExpiredLeasesLocked()

	// Find the peer with the highest surplus tokens (excluding the borrower)
	var bestLender *PeerQuota
	highestSurplus := 0

	for id, peer := range dmr.peers {
		if id == borrowerID {
			continue
		}
		surplus := peer.AvailableTokens - (peer.MaxCapacity / 4)
		if surplus > highestSurplus {
			highestSurplus = surplus
			bestLender = peer
		}
	}

	if bestLender == nil || highestSurplus <= 0 {
		return nil, errors.New("no peer available with surplus capacity for token lease")
	}

	tokensToGrant := requestedTokens
	if tokensToGrant > highestSurplus {
		tokensToGrant = highestSurplus
	}

	leaseID := "lease_" + borrowerID + "_" + time.Now().Format("150405.000")
	now := time.Now()

	bestLender.AvailableTokens -= tokensToGrant
	bestLender.LentTokens += tokensToGrant

	borrower, ok := dmr.peers[borrowerID]
	if ok {
		borrower.BorrowedTokens += tokensToGrant
		borrower.AvailableTokens += tokensToGrant
	}

	lease := &TokenLease{
		LeaseID:       leaseID,
		LenderPeerID:  bestLender.PeerID,
		BorrowerID:    borrowerID,
		TokensGranted: tokensToGrant,
		IssuedAt:      now,
		ExpiresAt:     now.Add(dmr.leaseTTL),
	}

	dmr.activeLeases[leaseID] = lease
	return lease, nil
}

// ReturnQuotaLease returns tokens from an active lease back to the lender.
func (dmr *DistributedMeshRebalancer) ReturnQuotaLease(leaseID string) error {
	dmr.mu.Lock()
	defer dmr.mu.Unlock()

	lease, exists := dmr.activeLeases[leaseID]
	if !exists {
		return errors.New("lease not found or already returned")
	}

	lender, ok := dmr.peers[lease.LenderPeerID]
	if ok {
		lender.AvailableTokens += lease.TokensGranted
		lender.LentTokens -= lease.TokensGranted
		if lender.LentTokens < 0 {
			lender.LentTokens = 0
		}
	}

	borrower, ok := dmr.peers[lease.BorrowerID]
	if ok {
		borrower.BorrowedTokens -= lease.TokensGranted
		if borrower.BorrowedTokens < 0 {
			borrower.BorrowedTokens = 0
		}
		borrower.AvailableTokens -= lease.TokensGranted
		if borrower.AvailableTokens < 0 {
			borrower.AvailableTokens = 0
		}
	}

	delete(dmr.activeLeases, leaseID)
	return nil
}

// GetTelemetry aggregates cluster quota distribution.
func (dmr *DistributedMeshRebalancer) GetTelemetry() MeshTelemetry {
	dmr.mu.Lock()
	defer dmr.mu.Unlock()

	dmr.cleanupExpiredLeasesLocked()

	totalCap := 0
	totalAvail := 0
	totalLent := 0
	peersList := make([]PeerQuota, 0, len(dmr.peers))

	for _, p := range dmr.peers {
		totalCap += p.MaxCapacity
		totalAvail += p.AvailableTokens
		totalLent += p.LentTokens
		peersList = append(peersList, *p)
	}

	saturation := 0.0
	if totalCap > 0 {
		used := totalCap - totalAvail
		if used < 0 {
			used = 0
		}
		saturation = (float64(used) / float64(totalCap)) * 100.0
	}

	return MeshTelemetry{
		TotalClusterCapacity: totalCap,
		TotalAvailableTokens: totalAvail,
		TotalLentTokens:      totalLent,
		ActiveLeaseCount:     len(dmr.activeLeases),
		ClusterSaturationPct: saturation,
		Peers:                peersList,
	}
}

func (dmr *DistributedMeshRebalancer) cleanupExpiredLeasesLocked() {
	now := time.Now()
	for id, lease := range dmr.activeLeases {
		if now.After(lease.ExpiresAt) {
			// Auto return expired tokens
			if lender, ok := dmr.peers[lease.LenderPeerID]; ok {
				lender.AvailableTokens += lease.TokensGranted
				lender.LentTokens -= lease.TokensGranted
				if lender.LentTokens < 0 {
					lender.LentTokens = 0
				}
			}
			if borrower, ok := dmr.peers[lease.BorrowerID]; ok {
				borrower.BorrowedTokens -= lease.TokensGranted
				if borrower.BorrowedTokens < 0 {
					borrower.BorrowedTokens = 0
				}
			}
			delete(dmr.activeLeases, id)
		}
	}
}

// HTTP Handlers for Mesh Rebalancer

type MeshLeaseRequest struct {
	BorrowerID      string `json:"borrower_id"`
	RequestedTokens int    `json:"requested_tokens"`
}

type MeshReturnRequest struct {
	LeaseID string `json:"lease_id"`
}

func (s *GatewayServer) handleMeshLease(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req MeshLeaseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
		return
	}

	lease, err := s.meshRebalancer.RequestQuotaLease(req.BorrowerID, req.RequestedTokens)
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(lease)
}

func (s *GatewayServer) handleMeshReturn(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req MeshReturnRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
		return
	}

	if err := s.meshRebalancer.ReturnQuotaLease(req.LeaseID); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "success",
		"message": "Quota lease returned successfully",
	})
}

func (s *GatewayServer) handleMeshTelemetry(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	telemetry := s.meshRebalancer.GetTelemetry()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(telemetry)
}
