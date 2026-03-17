package http

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/behnamebrahimi/neoprophet-go/internal/cmdstan"
	"github.com/behnamebrahimi/neoprophet-go/internal/diagnostics"
	"github.com/behnamebrahimi/neoprophet-go/internal/prophet"
)

// Server provides the HTTP REST API for Prophet.
type Server struct {
	runner *cmdstan.Runner
	models sync.Map // map[string]*prophet.Model
	mux    *http.ServeMux
}

// NewServer creates an HTTP server backed by the given Stan runner.
func NewServer(runner *cmdstan.Runner) *Server {
	s := &Server{runner: runner, mux: http.NewServeMux()}
	s.mux.HandleFunc("POST /v1/fit", s.handleFit)
	s.mux.HandleFunc("POST /v1/predict", s.handlePredict)
	s.mux.HandleFunc("POST /v1/cv", s.handleCV)
	s.mux.HandleFunc("GET /v1/health", s.handleHealth)
	return s
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	return s.mux
}

// --- Request/Response types ---

type FitRequest struct {
	DS     []float64     `json:"ds"`
	Y      []float64     `json:"y"`
	Cap    []float64     `json:"cap,omitempty"`
	Config prophet.Config `json:"config"`
}

type FitResponse struct {
	ModelID string              `json:"model_id"`
	Params  prophet.FittedParams `json:"params"`
}

type PredictRequest struct {
	ModelID string    `json:"model_id"`
	DS      []float64 `json:"ds,omitempty"`
	Periods int       `json:"periods,omitempty"`
	Freq    float64   `json:"freq,omitempty"` // seconds (default 86400 = 1 day)
}

type PredictResponse struct {
	Forecasts []prophet.Forecast `json:"forecasts"`
}

type CVRequest struct {
	DS          []float64     `json:"ds"`
	Y           []float64     `json:"y"`
	Config      prophet.Config `json:"config"`
	HorizonDays int           `json:"horizon_days"`
	PeriodDays  int           `json:"period_days"`
	InitialDays int           `json:"initial_days"`
}

type CVResponse struct {
	Folds   []diagnostics.CVResult `json:"folds"`
	Metrics diagnostics.Metrics    `json:"metrics"`
}

// --- Handlers ---

func (s *Server) handleFit(w http.ResponseWriter, r *http.Request) {
	var req FitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request: %v", err)
		return
	}

	if len(req.DS) != len(req.Y) {
		writeError(w, http.StatusBadRequest, "ds and y must have the same length")
		return
	}

	data := make([]prophet.DataPoint, len(req.DS))
	for i := range req.DS {
		dp := prophet.DataPoint{DS: req.DS[i], Y: req.Y[i]}
		if len(req.Cap) > i {
			dp.Cap = req.Cap[i]
		}
		data[i] = dp
	}

	f := prophet.NewForecaster(req.Config, s.runner)
	model, err := f.Fit(r.Context(), data)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "fit failed: %v", err)
		return
	}

	modelID := fmt.Sprintf("model-%d", hashDS(req.DS))
	s.models.Store(modelID, model)

	writeJSON(w, http.StatusOK, FitResponse{
		ModelID: modelID,
		Params:  model.Params,
	})
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request: %v", err)
		return
	}

	val, ok := s.models.Load(req.ModelID)
	if !ok {
		writeError(w, http.StatusNotFound, "model %q not found", req.ModelID)
		return
	}
	model := val.(*prophet.Model)

	ds := req.DS
	if len(ds) == 0 && req.Periods > 0 {
		freq := req.Freq
		if freq == 0 {
			freq = 86400 // daily
		}
		ds = prophet.MakeFutureDataframe(model, req.Periods, freq, false)
	}

	f := prophet.NewForecaster(model.Config, s.runner)
	forecasts := f.Predict(model, ds)

	writeJSON(w, http.StatusOK, PredictResponse{Forecasts: forecasts})
}

func (s *Server) handleCV(w http.ResponseWriter, r *http.Request) {
	var req CVRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request: %v", err)
		return
	}

	if len(req.DS) != len(req.Y) {
		writeError(w, http.StatusBadRequest, "ds and y must have the same length")
		return
	}

	data := make([]prophet.DataPoint, len(req.DS))
	for i := range req.DS {
		data[i] = prophet.DataPoint{DS: req.DS[i], Y: req.Y[i]}
	}

	f := prophet.NewForecaster(req.Config, s.runner)
	cvConfig := diagnostics.CVConfig{
		HorizonDays: req.HorizonDays,
		PeriodDays:  req.PeriodDays,
		InitialDays: req.InitialDays,
	}

	results, err := diagnostics.CrossValidate(r.Context(), f, data, cvConfig)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "cv failed: %v", err)
		return
	}

	metrics := diagnostics.ComputeMetrics(results)

	writeJSON(w, http.StatusOK, CVResponse{
		Folds:   results,
		Metrics: metrics,
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"healthy": true,
		"stan":    s.runner.ExePath,
	})
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("error encoding response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, format string, args ...any) {
	writeJSON(w, status, map[string]string{
		"error": fmt.Sprintf(format, args...),
	})
}

func hashDS(ds []float64) uint64 {
	var h uint64 = 14695981039346656037
	for _, v := range ds {
		h ^= uint64(v)
		h *= 1099511628211
	}
	return h
}

// ListenAndServe starts the HTTP server. Blocks until ctx is cancelled.
func ListenAndServe(ctx context.Context, addr string, runner *cmdstan.Runner) error {
	srv := NewServer(runner)
	httpSrv := &http.Server{
		Addr:    addr,
		Handler: srv.Handler(),
	}

	go func() {
		<-ctx.Done()
		httpSrv.Close()
	}()

	log.Printf("HTTP server listening on %s", addr)
	return httpSrv.ListenAndServe()
}
