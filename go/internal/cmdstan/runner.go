package cmdstan

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Runner wraps a compiled CmdStan binary and manages invocations.
type Runner struct {
	ExePath string
	Timeout time.Duration
}

// NewRunner creates a Runner pointing at a compiled Stan model binary.
func NewRunner(exePath string) *Runner {
	return &Runner{
		ExePath: exePath,
		Timeout: 5 * time.Minute,
	}
}

// OptimizeResult holds the output of a Stan optimization run.
type OptimizeResult struct {
	Params map[string][]float64
	LP     float64
}

// Optimize runs MAP estimation (method=optimize) on the Stan model.
func (r *Runner) Optimize(ctx context.Context, data, init map[string]any, algorithm string) (*OptimizeResult, error) {
	return r.runInTempDir(ctx, func(dir string) (*OptimizeResult, error) {
		dataFile := filepath.Join(dir, "data.json")
		if err := writeJSON(dataFile, data); err != nil {
			return nil, fmt.Errorf("writing data: %w", err)
		}

		initFile := filepath.Join(dir, "init.json")
		if err := writeJSON(initFile, init); err != nil {
			return nil, fmt.Errorf("writing init: %w", err)
		}

		outputFile := filepath.Join(dir, "output.csv")

		ctx, cancel := context.WithTimeout(ctx, r.Timeout)
		defer cancel()

		cmd := exec.CommandContext(ctx, r.ExePath,
			"method=optimize",
			fmt.Sprintf("algorithm=%s", algorithm),
			"iter=10000",
			"data", fmt.Sprintf("file=%s", dataFile),
			fmt.Sprintf("init=%s", initFile),
			"output", fmt.Sprintf("file=%s", outputFile),
		)

		var stderr strings.Builder
		cmd.Stderr = &stderr

		if err := cmd.Run(); err != nil {
			return nil, fmt.Errorf("cmdstan optimize (%s) failed: %w\nstderr: %s", algorithm, err, stderr.String())
		}

		params, err := ParseStanCSV(outputFile)
		if err != nil {
			return nil, fmt.Errorf("parsing output: %w", err)
		}

		result := &OptimizeResult{Params: make(map[string][]float64)}
		for k, v := range params {
			if k == "lp__" {
				if len(v) > 0 {
					result.LP = v[0]
				}
				continue
			}
			result.Params[k] = v
		}
		return result, nil
	})
}

// OptimizeWithFallback tries L-BFGS (or Newton for small T), falls back to Newton on failure.
func (r *Runner) OptimizeWithFallback(ctx context.Context, data, init map[string]any) (*OptimizeResult, error) {
	t, _ := data["T"].(int)
	algo := "lbfgs"
	if t < 100 {
		algo = "newton"
	}

	result, err := r.Optimize(ctx, data, init, algo)
	if err != nil && algo == "lbfgs" {
		result, err = r.Optimize(ctx, data, init, "newton")
	}
	return result, err
}

// Sample runs MCMC sampling on the Stan model.
func (r *Runner) Sample(ctx context.Context, data, init map[string]any, chainID, numSamples int) (*OptimizeResult, error) {
	return r.runInTempDir(ctx, func(dir string) (*OptimizeResult, error) {
		dataFile := filepath.Join(dir, "data.json")
		if err := writeJSON(dataFile, data); err != nil {
			return nil, fmt.Errorf("writing data: %w", err)
		}

		initFile := filepath.Join(dir, "init.json")
		if err := writeJSON(initFile, init); err != nil {
			return nil, fmt.Errorf("writing init: %w", err)
		}

		outputFile := filepath.Join(dir, "output.csv")
		warmup := numSamples / 2
		sampling := numSamples - warmup

		ctx, cancel := context.WithTimeout(ctx, r.Timeout)
		defer cancel()

		cmd := exec.CommandContext(ctx, r.ExePath,
			"method=sample",
			fmt.Sprintf("num_warmup=%d", warmup),
			fmt.Sprintf("num_samples=%d", sampling),
			fmt.Sprintf("id=%d", chainID+1),
			"data", fmt.Sprintf("file=%s", dataFile),
			fmt.Sprintf("init=%s", initFile),
			"output", fmt.Sprintf("file=%s", outputFile),
		)

		var stderr strings.Builder
		cmd.Stderr = &stderr

		if err := cmd.Run(); err != nil {
			return nil, fmt.Errorf("cmdstan sample (chain %d) failed: %w\nstderr: %s", chainID, err, stderr.String())
		}

		params, err := ParseStanCSV(outputFile)
		if err != nil {
			return nil, fmt.Errorf("parsing output: %w", err)
		}

		result := &OptimizeResult{Params: make(map[string][]float64)}
		for k, v := range params {
			if strings.HasSuffix(k, "__") {
				continue // skip diagnostic columns
			}
			result.Params[k] = v
		}
		return result, nil
	})
}

func (r *Runner) runInTempDir(ctx context.Context, fn func(dir string) (*OptimizeResult, error)) (*OptimizeResult, error) {
	dir, err := os.MkdirTemp("", "cmdstan-*")
	if err != nil {
		return nil, fmt.Errorf("creating temp dir: %w", err)
	}
	defer os.RemoveAll(dir)
	return fn(dir)
}

func writeJSON(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(v)
}
