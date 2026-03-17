package pool

import (
	"context"
	"runtime"
	"sync"

	"golang.org/x/sync/errgroup"
)

// StanPool manages bounded-concurrency execution of CmdStan invocations.
type StanPool struct {
	MaxWorkers int
}

// NewStanPool creates a worker pool. If maxWorkers <= 0, uses runtime.NumCPU().
func NewStanPool(maxWorkers int) *StanPool {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}
	return &StanPool{MaxWorkers: maxWorkers}
}

// Result holds the output of a single parallel job.
type Result[T any] struct {
	Index int
	Value T
}

// RunParallel executes jobs concurrently with bounded parallelism.
// Each job receives its index and returns a result or error.
// Results are returned in input order.
func RunParallel[T any](ctx context.Context, pool *StanPool, n int, job func(ctx context.Context, i int) (T, error)) ([]T, error) {
	results := make([]T, n)
	var mu sync.Mutex

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(pool.MaxWorkers)

	for i := 0; i < n; i++ {
		i := i
		g.Go(func() error {
			val, err := job(ctx, i)
			if err != nil {
				return err
			}
			mu.Lock()
			results[i] = val
			mu.Unlock()
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}
	return results, nil
}
