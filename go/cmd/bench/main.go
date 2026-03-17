package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
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

func main() {
	stanPath := os.Getenv("PROPHET_STAN_BINARY")
	if stanPath == "" {
		fmt.Fprintln(os.Stderr, "set PROPHET_STAN_BINARY env var to compiled prophet model")
		os.Exit(1)
	}

	nDays := 1095
	if len(os.Args) > 1 {
		fmt.Sscan(os.Args[1], &nDays)
	}

	fmt.Printf("=== Go Prophet Benchmark (%d days) ===\n\n", nDays)

	data := generateDataset(nDays)
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
		InitialDays: 365,
		PeriodDays:  90,
		HorizonDays: 90,
		MaxWorkers:  0, // use all CPUs
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

// generateDataset creates the same synthetic data as the Python benchmark.
func generateDataset(nDays int) []prophet.DataPoint {
	rng := rand.New(rand.NewSource(42))
	baseDS := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)

	data := make([]prophet.DataPoint, nDays)
	for i := 0; i < nDays; i++ {
		t := float64(i)
		ds := baseDS.Add(time.Duration(i) * 24 * time.Hour)

		trend := 100 + 0.05*t
		yearly := 10*math.Sin(2*math.Pi*t/365.25) +
			5*math.Cos(2*math.Pi*t/365.25) +
			3*math.Sin(4*math.Pi*t/365.25)
		weekly := 5 * math.Sin(2*math.Pi*t/7)
		noise := rng.NormFloat64() * 3

		data[i] = prophet.DataPoint{
			DS: float64(ds.Unix()),
			Y:  trend + yearly + weekly + noise,
		}
	}
	return data
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
