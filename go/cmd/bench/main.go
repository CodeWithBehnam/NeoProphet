package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/behnamebrahimi/neoprophet-go/internal/cmdstan"
	"github.com/behnamebrahimi/neoprophet-go/internal/diagnostics"
	"github.com/behnamebrahimi/neoprophet-go/internal/prophet"
)

type benchResults struct {
	Language     string  `json:"language"`
	NDays        int     `json:"n_days"`
	FitMeanS     float64 `json:"fit_mean_s"`
	PredictMeanS float64 `json:"predict_mean_s"`
	PredictNRows int     `json:"predict_n_rows"`
	CVTimeS      float64 `json:"cv_time_s"`
	CVCutoffs    int     `json:"cv_cutoffs"`
	CVMAPE       float64 `json:"cv_mape"`
}

type dataRecord struct {
	DS float64 `json:"ds"`
	Y  float64 `json:"y"`
}

func main() {
	stanPath := os.Getenv("PROPHET_STAN_BINARY")
	if stanPath == "" {
		fmt.Fprintln(os.Stderr, "set PROPHET_STAN_BINARY env var to compiled prophet model")
		os.Exit(1)
	}

	// Load dataset exported by Python benchmark for exact parity
	dataPath := "/tmp/prophet_bench_data.json"
	data, err := loadDataset(dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load %s: %v\nRun the Python benchmark first.\n", dataPath, err)
		os.Exit(1)
	}

	nDays := len(data)
	fmt.Printf("=== Go Prophet Benchmark (%d days) ===\n", nDays)
	fmt.Printf("Loaded dataset from %s\n\n", dataPath)

	runner := cmdstan.NewRunner(stanPath)
	config := prophet.DefaultConfig()
	ctx := context.Background()

	// --- Fit ---
	fmt.Println("--- Fit ---")
	nFitRuns := 3
	fitTimes := make([]float64, nFitRuns)
	var model *prophet.Model
	f := prophet.NewForecaster(config, runner)

	for i := 0; i < nFitRuns; i++ {
		start := time.Now()
		m, err := f.Fit(ctx, data)
		elapsed := time.Since(start).Seconds()
		if err != nil {
			fmt.Fprintf(os.Stderr, "  fit run %d failed: %v\n", i+1, err)
			os.Exit(1)
		}
		fitTimes[i] = elapsed
		fmt.Printf("  fit run %d/%d: %.3fs\n", i+1, nFitRuns, elapsed)
		if model == nil {
			model = m
		}
	}
	fitMean := mean(fitTimes)

	// --- Predict ---
	fmt.Println("\n--- Predict (365 days ahead) ---")
	periods := 365
	futureDS := prophet.MakeFutureDataframe(model, periods, 86400, true)
	nPredRuns := 5
	predTimes := make([]float64, nPredRuns)

	for i := 0; i < nPredRuns; i++ {
		start := time.Now()
		_ = f.Predict(model, futureDS)
		elapsed := time.Since(start).Seconds()
		predTimes[i] = elapsed
		fmt.Printf("  predict run %d/%d: %.3fs\n", i+1, nPredRuns, elapsed)
	}
	predMean := mean(predTimes)

	// --- Cross-Validation ---
	fmt.Println("\n--- Cross-Validation ---")
	cvStart := time.Now()
	cvResults, err := diagnostics.CrossValidate(ctx, f, data, diagnostics.CVConfig{
		InitialDays: 730,
		PeriodDays:  90,
		HorizonDays: 90,
		MaxWorkers:  0,
	})
	cvTime := time.Since(cvStart).Seconds()
	if err != nil {
		fmt.Fprintf(os.Stderr, "cv failed: %v\n", err)
		os.Exit(1)
	}

	metrics := diagnostics.ComputeMetrics(cvResults)
	fmt.Printf("  cv: %.3fs (%d cutoffs, MAPE=%.4f)\n", cvTime, len(cvResults), metrics.MAPE)

	// --- Summary ---
	results := benchResults{
		Language:     "go",
		NDays:        nDays,
		FitMeanS:     fitMean,
		PredictMeanS: predMean,
		PredictNRows: len(futureDS),
		CVTimeS:      cvTime,
		CVCutoffs:    len(cvResults),
		CVMAPE:       metrics.MAPE,
	}

	fmt.Println("\n=== Summary ===")
	fmt.Printf("  language: %s\n", results.Language)
	fmt.Printf("  n_days: %d\n", results.NDays)
	fmt.Printf("  fit_mean_s: %.4f\n", results.FitMeanS)
	fmt.Printf("  predict_mean_s: %.4f\n", results.PredictMeanS)
	fmt.Printf("  predict_n_rows: %d\n", results.PredictNRows)
	fmt.Printf("  cv_time_s: %.4f\n", results.CVTimeS)
	fmt.Printf("  cv_cutoffs: %d\n", results.CVCutoffs)
	fmt.Printf("  cv_mape: %.4f\n", results.CVMAPE)

	out, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("/tmp/prophet_bench_go.json", out, 0644)
	fmt.Println("\nResults saved to /tmp/prophet_bench_go.json")
}

func loadDataset(path string) ([]prophet.DataPoint, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var records []dataRecord
	if err := json.Unmarshal(raw, &records); err != nil {
		return nil, err
	}

	data := make([]prophet.DataPoint, len(records))
	for i, r := range records {
		data[i] = prophet.DataPoint{DS: r.DS, Y: r.Y}
	}
	return data, nil
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}
