package prophet

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/behnamebrahimi/neoprophet-go/internal/cmdstan"
)

// FittedParams holds the parameters estimated by Stan.
type FittedParams struct {
	K        float64   `json:"k"`
	M        float64   `json:"m"`
	Delta    []float64 `json:"delta"`
	Beta     []float64 `json:"beta"`
	SigmaObs float64  `json:"sigma_obs"`
}

// Model represents a fitted Prophet model.
type Model struct {
	Config       Config            `json:"config"`
	Params       FittedParams      `json:"params"`
	Scale        ScaleParams       `json:"scale"`
	Changepoints []float64         `json:"changepoints_t"` // normalized changepoint times
	Seasonalities []SeasonalitySpec `json:"seasonalities"`
	Holidays     []HolidayEntry    `json:"holidays"`
	TDays        []float64         `json:"-"` // training time in days
}

// DataPoint represents a single observation in the time series.
type DataPoint struct {
	DS    float64 // unix timestamp in seconds
	Y     float64
	Cap   float64 // for logistic growth
	Floor float64 // for logistic growth
}

// Forecast holds prediction results for a single time point.
type Forecast struct {
	DS        float64 `json:"ds"`
	Trend     float64 `json:"trend"`
	Yhat      float64 `json:"yhat"`
	YhatLower float64 `json:"yhat_lower"`
	YhatUpper float64 `json:"yhat_upper"`
}

// Forecaster provides the Fit/Predict API.
type Forecaster struct {
	config Config
	runner *cmdstan.Runner
}

// NewForecaster creates a Forecaster with the given config and Stan runner.
func NewForecaster(config Config, runner *cmdstan.Runner) *Forecaster {
	return &Forecaster{config: config, runner: runner}
}

// Fit trains a Prophet model on the given data.
func (f *Forecaster) Fit(ctx context.Context, data []DataPoint) (*Model, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("need at least 2 data points, got %d", len(data))
	}

	// Sort by timestamp
	sort.Slice(data, func(i, j int) bool { return data[i].DS < data[j].DS })

	// Extract y and ds
	ds := make([]float64, len(data))
	y := make([]float64, len(data))
	caps := make([]float64, len(data))
	for i, d := range data {
		ds[i] = d.DS
		y[i] = d.Y
		caps[i] = d.Cap
	}

	// Normalize time to [0, 1]
	t, tStart, tScale := NormalizeTime(ds)

	// Scale y
	yScaled, scaleParams := ScaleY(y, f.config.Scaling)
	scaleParams.TStart = tStart
	scaleParams.TScale = tScale

	// Convert t to days for seasonality features
	tDays := make([]float64, len(t))
	for i, v := range ds {
		tDays[i] = v / 86400.0 // seconds to days
	}

	// Select changepoints
	nCp := f.config.NChangepoints
	if nCp >= len(t) {
		nCp = len(t) - 1
	}
	cpTimes := SelectChangepoints(t, nCp, f.config.ChangepointRange)

	// Build seasonality specs based on auto-detection
	specs := f.autoSeasonalities(tDays)

	// Build seasonality features
	X, K, sA, sM, sigmas := MakeSeasonalityFeatures(tDays, specs)
	if K == 0 {
		// At minimum we need 1 column for Stan
		K = 1
		X = make([]float64, len(t))
		sA = []float64{1.0}
		sM = []float64{0.0}
		sigmas = []float64{f.config.SeasonalityPriorScale}
	}

	// Stan builds its own changepoint matrix internally
	S := len(cpTimes)

	// Prepare cap for Stan
	capScaled := make([]float64, len(t))
	if f.config.Growth == GrowthLogistic {
		capScaled = ScaleLogisticCap(caps, scaleParams.Floor, scaleParams.YScale)
	} else {
		for i := range capScaled {
			capScaled[i] = 0.0
		}
	}

	// Trend indicator
	trendInd := 0
	switch f.config.Growth {
	case GrowthLogistic:
		trendInd = 1
	case GrowthFlat:
		trendInd = 2
	}

	// Build X as 2D array for JSON
	X2D := make([][]float64, len(t))
	for i := range X2D {
		row := make([]float64, K)
		for j := 0; j < K; j++ {
			row[j] = X[i*K+j]
		}
		X2D[i] = row
	}

	stanData := map[string]any{
		"T":               len(t),
		"S":               S,
		"K":               K,
		"tau":             f.config.ChangepointPriorScale,
		"trend_indicator": trendInd,
		"y":               yScaled,
		"t":               t,
		"cap":             capScaled,
		"t_change":        cpTimes,
		"s_a":             sA,
		"s_m":             sM,
		"X":               X2D,
		"sigmas":          sigmas,
	}

	// Initial params — match Python: line through first and last points
	initK, initM := LinearGrowthInit(t, yScaled)
	stanInit := map[string]any{
		"k":         initK,
		"m":         initM,
		"delta":     makeZeros(S),
		"beta":      makeZeros(K),
		"sigma_obs": 1.0,
	}

	// Fit via Stan
	result, err := f.runner.OptimizeWithFallback(ctx, stanData, stanInit)
	if err != nil {
		return nil, fmt.Errorf("stan fit: %w", err)
	}

	// Parse results
	grouped := cmdstan.GroupParams(result.Params, 1)
	params := FittedParams{
		SigmaObs: getScalar(grouped, "sigma_obs"),
		K:        getScalar(grouped, "k"),
		M:        getScalar(grouped, "m"),
		Delta:    grouped["delta"],
		Beta:     grouped["beta"],
	}

	model := &Model{
		Config:        f.config,
		Params:        params,
		Scale:         scaleParams,
		Changepoints:  cpTimes,
		Seasonalities: specs,
		TDays:         tDays,
	}

	return model, nil
}

// Predict generates forecasts from a fitted model.
func (f *Forecaster) Predict(model *Model, ds []float64) []Forecast {
	T := len(ds)

	// Normalize time using training scale
	t := make([]float64, T)
	for i, v := range ds {
		t[i] = (v - model.Scale.TStart) / model.Scale.TScale
	}

	// Time in days for seasonality
	tDays := make([]float64, T)
	for i, v := range ds {
		tDays[i] = v / 86400.0
	}

	// Compute trend
	A := MakeChangepointMatrix(t, model.Changepoints)
	var trend []float64

	switch model.Config.Growth {
	case GrowthLogistic:
		cap := make([]float64, T)
		for i := range cap {
			cap[i] = 1.0 // placeholder; real usage passes actual caps
		}
		trend = PiecewiseLogistic(model.Params.K, model.Params.M, model.Params.Delta, t, cap, A, model.Changepoints)
	case GrowthFlat:
		trend = FlatTrend(model.Params.M, T)
	default:
		trend = PiecewiseLinear(model.Params.K, model.Params.M, model.Params.Delta, t, A, model.Changepoints)
	}

	// Compute seasonal components
	X, K, sA, sM, _ := MakeSeasonalityFeatures(tDays, model.Seasonalities)

	forecasts := make([]Forecast, T)
	for i := 0; i < T; i++ {
		// Compute additive and multiplicative terms
		var addTerms, multTerms float64
		if K > 0 && len(model.Params.Beta) == K {
			for j := 0; j < K; j++ {
				val := X[i*K+j] * model.Params.Beta[j]
				addTerms += sA[j] * val
				multTerms += sM[j] * val
			}
		}

		// Unscale: Python does trend * y_scale + floor
		// For absmax: floor=0. For minmax: floor=y_min.
		floor := model.Scale.Floor
		if model.Config.Scaling == "minmax" {
			floor = model.Scale.YMin
		}
		trendUnscaled := trend[i]*model.Scale.YScale + floor

		// yhat = trend * (1 + mult) + add * y_scale
		yhat := trendUnscaled*(1+multTerms) + addTerms*model.Scale.YScale

		forecasts[i] = Forecast{
			DS:    ds[i],
			Trend: trendUnscaled,
			Yhat:  yhat,
		}
	}

	return forecasts
}

// MakeFutureDataframe generates timestamps for future predictions.
// periods is the number of future steps, freqSeconds is the step size
// (86400 for daily). If includeHistory is true, training timestamps are prepended.
func MakeFutureDataframe(model *Model, periods int, freqSeconds float64, includeHistory bool) []float64 {
	// Recover training timestamps from normalized values
	var histDS []float64
	if includeHistory {
		histDS = make([]float64, len(model.TDays))
		for i, d := range model.TDays {
			histDS[i] = d * 86400.0
		}
	}

	lastDS := model.Scale.TStart + model.Scale.TScale
	future := make([]float64, periods)
	for i := 0; i < periods; i++ {
		future[i] = lastDS + freqSeconds*float64(i+1)
	}

	return append(histDS, future...)
}

func (f *Forecaster) autoSeasonalities(tDays []float64) []SeasonalitySpec {
	if len(tDays) < 2 {
		return nil
	}

	rangeDays := tDays[len(tDays)-1] - tDays[0]
	var specs []SeasonalitySpec

	// Yearly: need > 2 years of data, use 10 Fourier terms
	if rangeDays >= 730 {
		specs = append(specs, SeasonalitySpec{
			Name: "yearly", Period: 365.25, FourierOrder: 10,
			PriorScale: f.config.SeasonalityPriorScale, Mode: f.config.SeasonalityMode,
		})
	}

	// Weekly: need > 2 weeks of data, use 3 Fourier terms
	if rangeDays >= 14 {
		specs = append(specs, SeasonalitySpec{
			Name: "weekly", Period: 7.0, FourierOrder: 3,
			PriorScale: f.config.SeasonalityPriorScale, Mode: f.config.SeasonalityMode,
		})
	}

	// Daily: need sub-daily data (check median interval < 1 day)
	if len(tDays) > 2 {
		intervals := make([]float64, len(tDays)-1)
		for i := 1; i < len(tDays); i++ {
			intervals[i-1] = tDays[i] - tDays[i-1]
		}
		sort.Float64s(intervals)
		median := intervals[len(intervals)/2]
		if median < 1.0 && rangeDays >= 2 {
			specs = append(specs, SeasonalitySpec{
				Name: "daily", Period: 1.0, FourierOrder: 4,
				PriorScale: f.config.SeasonalityPriorScale, Mode: f.config.SeasonalityMode,
			})
		}
	}

	return specs
}

// LinearGrowthInit computes initial k (rate) and m (offset) matching Python Prophet.
// Uses the first and last data points so the line passes through both.
func LinearGrowthInit(t, yScaled []float64) (k, m float64) {
	n := len(t)
	if n < 2 {
		return 0, 0
	}
	// Python: i0 = ds.idxmin(), i1 = ds.idxmax() — first and last by time
	// Since data is sorted, i0=0, i1=n-1
	T := t[n-1] - t[0]
	if math.Abs(T) < 1e-12 {
		return 0, yScaled[0]
	}
	k = (yScaled[n-1] - yScaled[0]) / T
	m = yScaled[0] - k*t[0]
	return k, m
}

func makeZeros(n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	return make([]float64, n)
}

func getScalar(params map[string][]float64, key string) float64 {
	vals, ok := params[key]
	if !ok || len(vals) == 0 {
		return 0
	}
	return vals[0]
}
