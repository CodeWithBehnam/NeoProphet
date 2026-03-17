package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/behnamebrahimi/neoprophet-go/internal/cmdstan"
	"github.com/behnamebrahimi/neoprophet-go/internal/prophet"
)

// Request is the JSON input from the notebook.
type Request struct {
	DataPath              string  `json:"data_path"`
	Name                  string  `json:"name"`
	ChangepointPriorScale float64 `json:"changepoint_prior_scale"`
	SeasonalityPriorScale float64 `json:"seasonality_prior_scale"`
	SeasonalityMode       string  `json:"seasonality_mode"`
	Growth                string  `json:"growth"`
	NChangepoints         int     `json:"n_changepoints"`
	PredictPeriods        int     `json:"predict_periods"`
}

// Result is the JSON output back to the notebook.
type Result struct {
	Name         string  `json:"name"`
	NDays        int     `json:"n_days"`
	FitTimeS     float64 `json:"fit_time_s"`
	PredictTimeS float64 `json:"predict_time_s"`
	PredictRows  int     `json:"predict_rows"`
}

type dataRecord struct {
	DS float64 `json:"ds"`
	Y  float64 `json:"y"`
}

func main() {
	stanPath := os.Getenv("PROPHET_STAN_BINARY")
	if stanPath == "" {
		fatal("set PROPHET_STAN_BINARY env var")
	}

	if len(os.Args) < 2 {
		fatal("usage: bench_all '<json request>'")
	}

	var req Request
	if err := json.Unmarshal([]byte(os.Args[1]), &req); err != nil {
		fatal("invalid JSON: %v", err)
	}

	data, err := loadDataset(req.DataPath)
	if err != nil {
		fatal("load %s: %v", req.DataPath, err)
	}

	config := prophet.DefaultConfig()
	if req.ChangepointPriorScale > 0 {
		config.ChangepointPriorScale = req.ChangepointPriorScale
	}
	if req.SeasonalityPriorScale > 0 {
		config.SeasonalityPriorScale = req.SeasonalityPriorScale
	}
	if req.SeasonalityMode != "" {
		config.SeasonalityMode = prophet.SeasonalityMode(req.SeasonalityMode)
	}
	if req.Growth != "" {
		config.Growth = prophet.Growth(req.Growth)
	}
	if req.NChangepoints > 0 {
		config.NChangepoints = req.NChangepoints
	}

	runner := cmdstan.NewRunner(stanPath)
	f := prophet.NewForecaster(config, runner)
	ctx := context.Background()

	// Fit
	fitStart := time.Now()
	model, err := f.Fit(ctx, data)
	fitTime := time.Since(fitStart).Seconds()
	if err != nil {
		fatal("fit %s: %v", req.Name, err)
	}

	// Predict
	periods := req.PredictPeriods
	if periods <= 0 {
		periods = 90
	}
	futureDS := prophet.MakeFutureDataframe(model, periods, 86400, false)

	predStart := time.Now()
	forecasts := f.Predict(model, futureDS)
	predTime := time.Since(predStart).Seconds()

	result := Result{
		Name:         req.Name,
		NDays:        len(data),
		FitTimeS:     fitTime,
		PredictTimeS: predTime,
		PredictRows:  len(forecasts),
	}

	out, _ := json.Marshal(result)
	fmt.Println(string(out))
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

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
