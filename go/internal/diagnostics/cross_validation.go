package diagnostics

import (
	"context"
	"fmt"
	"sort"

	"github.com/behnamebrahimi/neoprophet-go/internal/pool"
	"github.com/behnamebrahimi/neoprophet-go/internal/prophet"
)

// CVResult holds the cross-validation output for a single fold.
type CVResult struct {
	Cutoff float64   `json:"cutoff"` // unix timestamp
	DS     []float64 `json:"ds"`
	Yhat   []float64 `json:"yhat"`
	Y      []float64 `json:"y"`
}

// CVConfig holds parameters for cross-validation in days.
type CVConfig struct {
	HorizonDays int // forecast horizon in days
	PeriodDays  int // days between cutoffs (default: horizon/2)
	InitialDays int // initial training window in days (default: 3*horizon)
	MaxWorkers  int // concurrency limit (0 = NumCPU)
}

// Fitter can fit and predict a Prophet model.
type Fitter interface {
	Fit(ctx context.Context, data []prophet.DataPoint) (*prophet.Model, error)
	Predict(model *prophet.Model, ds []float64) []prophet.Forecast
}

const secondsPerDay = 86400.0

// CrossValidate runs simulated historical forecasts in parallel.
func CrossValidate(
	ctx context.Context,
	fitter Fitter,
	data []prophet.DataPoint,
	cvConfig CVConfig,
) ([]CVResult, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("need at least 2 data points")
	}

	// Sort data by timestamp
	sorted := make([]prophet.DataPoint, len(data))
	copy(sorted, data)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].DS < sorted[j].DS })

	horizon := float64(cvConfig.HorizonDays) * secondsPerDay

	periodDays := cvConfig.PeriodDays
	if periodDays == 0 {
		periodDays = cvConfig.HorizonDays / 2
		if periodDays == 0 {
			periodDays = 1
		}
	}
	period := float64(periodDays) * secondsPerDay

	initialDays := cvConfig.InitialDays
	if initialDays == 0 {
		initialDays = cvConfig.HorizonDays * 3
	}
	initial := float64(initialDays) * secondsPerDay

	// Generate cutoffs
	cutoffs := generateCutoffs(sorted, horizon, initial, period)
	if len(cutoffs) == 0 {
		return nil, fmt.Errorf("no valid cutoffs found: not enough data for horizon=%dd, initial=%dd",
			cvConfig.HorizonDays, initialDays)
	}

	wp := pool.NewStanPool(cvConfig.MaxWorkers)

	results, err := pool.RunParallel(ctx, wp, len(cutoffs), func(ctx context.Context, i int) (CVResult, error) {
		cutoff := cutoffs[i]

		// Split data at cutoff
		var train []prophet.DataPoint
		var test []prophet.DataPoint
		for _, d := range sorted {
			if d.DS <= cutoff {
				train = append(train, d)
			} else if d.DS <= cutoff+horizon {
				test = append(test, d)
			}
		}

		if len(train) < 2 || len(test) == 0 {
			return CVResult{Cutoff: cutoff}, nil
		}

		// Fit model on training data
		model, err := fitter.Fit(ctx, train)
		if err != nil {
			return CVResult{}, fmt.Errorf("fold cutoff=%.0f fit: %w", cutoff, err)
		}

		// Predict on test dates
		testDS := make([]float64, len(test))
		testY := make([]float64, len(test))
		for j, d := range test {
			testDS[j] = d.DS
			testY[j] = d.Y
		}

		forecasts := fitter.Predict(model, testDS)

		yhat := make([]float64, len(forecasts))
		for j, fc := range forecasts {
			yhat[j] = fc.Yhat
		}

		return CVResult{
			Cutoff: cutoff,
			DS:     testDS,
			Yhat:   yhat,
			Y:      testY,
		}, nil
	})

	if err != nil {
		return nil, err
	}

	return results, nil
}

// generateCutoffs produces the list of cutoff timestamps for cross-validation.
func generateCutoffs(data []prophet.DataPoint, horizon, initial, period float64) []float64 {
	if len(data) == 0 {
		return nil
	}

	minDS := data[0].DS
	maxDS := data[len(data)-1].DS

	var cutoffs []float64
	cutoff := maxDS - horizon

	for cutoff >= minDS+initial {
		cutoffs = append(cutoffs, cutoff)
		cutoff -= period
	}

	sort.Float64s(cutoffs)
	return cutoffs
}
