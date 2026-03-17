package prophet

// Growth mode for the trend component.
type Growth string

const (
	GrowthLinear   Growth = "linear"
	GrowthLogistic Growth = "logistic"
	GrowthFlat     Growth = "flat"
)

// SeasonalityMode controls how seasonality interacts with trend.
type SeasonalityMode string

const (
	ModeAdditive       SeasonalityMode = "additive"
	ModeMultiplicative SeasonalityMode = "multiplicative"
)

// Config holds all Prophet hyperparameters, mirroring the Python Prophet.__init__ signature.
type Config struct {
	Growth                 Growth          `json:"growth"`
	NChangepoints          int             `json:"n_changepoints"`
	ChangepointRange       float64         `json:"changepoint_range"`
	YearlySeasonality      any             `json:"yearly_seasonality"`  // "auto" or int
	WeeklySeasonality      any             `json:"weekly_seasonality"`  // "auto" or int
	DailySeasonality       any             `json:"daily_seasonality"`   // "auto" or int
	SeasonalityMode        SeasonalityMode `json:"seasonality_mode"`
	SeasonalityPriorScale  float64         `json:"seasonality_prior_scale"`
	ChangepointPriorScale  float64         `json:"changepoint_prior_scale"`
	HolidaysPriorScale     float64         `json:"holidays_prior_scale"`
	MCMCSamples            int             `json:"mcmc_samples"`
	IntervalWidth          float64         `json:"interval_width"`
	UncertaintySamples     int             `json:"uncertainty_samples"`
	Scaling                string          `json:"scaling"` // "absmax" or "minmax"
}

// DefaultConfig returns a Config with Prophet's default hyperparameters.
func DefaultConfig() Config {
	return Config{
		Growth:                 GrowthLinear,
		NChangepoints:          25,
		ChangepointRange:       0.8,
		YearlySeasonality:      "auto",
		WeeklySeasonality:      "auto",
		DailySeasonality:       "auto",
		SeasonalityMode:        ModeAdditive,
		SeasonalityPriorScale:  10.0,
		ChangepointPriorScale:  0.05,
		HolidaysPriorScale:     10.0,
		MCMCSamples:            0,
		IntervalWidth:          0.80,
		UncertaintySamples:     1000,
		Scaling:                "absmax",
	}
}

// SeasonalitySpec describes a single seasonal component.
type SeasonalitySpec struct {
	Name          string          `json:"name"`
	Period        float64         `json:"period"`         // days
	FourierOrder  int             `json:"fourier_order"`
	PriorScale    float64         `json:"prior_scale"`
	Mode          SeasonalityMode `json:"mode"`
	ConditionName string          `json:"condition_name"` // optional boolean column
}

// HolidayEntry represents a single holiday occurrence.
type HolidayEntry struct {
	Holiday    string  `json:"holiday"`
	DS         float64 `json:"ds"`          // unix timestamp
	LowerWindow int    `json:"lower_window"` // days before
	UpperWindow int    `json:"upper_window"` // days after
}
