package cmdstan

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseStanCSV(t *testing.T) {
	content := `# CmdStan output
# model = prophet
lp__,k,m,delta.1,delta.2,sigma_obs,beta.1,beta.2,beta.3
-123.45,0.5,1.2,0.01,-0.02,0.15,0.3,0.1,-0.05
`
	dir := t.TempDir()
	path := filepath.Join(dir, "output.csv")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	raw, err := ParseStanCSV(path)
	if err != nil {
		t.Fatalf("ParseStanCSV failed: %v", err)
	}

	// Check scalar params
	assertFloat(t, raw, "lp__", 0, -123.45)
	assertFloat(t, raw, "k", 0, 0.5)
	assertFloat(t, raw, "m", 0, 1.2)
	assertFloat(t, raw, "sigma_obs", 0, 0.15)

	// Check indexed params
	assertFloat(t, raw, "delta.1", 0, 0.01)
	assertFloat(t, raw, "delta.2", 0, -0.02)
	assertFloat(t, raw, "beta.1", 0, 0.3)
	assertFloat(t, raw, "beta.2", 0, 0.1)
	assertFloat(t, raw, "beta.3", 0, -0.05)
}

func TestGroupParams(t *testing.T) {
	raw := map[string][]float64{
		"lp__":      {-123.45},
		"k":         {0.5},
		"m":         {1.2},
		"delta.1":   {0.01},
		"delta.2":   {-0.02},
		"sigma_obs": {0.15},
		"beta.1":    {0.3},
		"beta.2":    {0.1},
		"beta.3":    {-0.05},
	}

	grouped := GroupParams(raw, 1)

	// Scalars preserved
	assertFloat(t, grouped, "k", 0, 0.5)
	assertFloat(t, grouped, "m", 0, 1.2)
	assertFloat(t, grouped, "sigma_obs", 0, 0.15)

	// Indexed grouped
	if len(grouped["delta"]) != 2 {
		t.Fatalf("expected delta length 2, got %d", len(grouped["delta"]))
	}
	assertFloat(t, grouped, "delta", 0, 0.01)
	assertFloat(t, grouped, "delta", 1, -0.02)

	if len(grouped["beta"]) != 3 {
		t.Fatalf("expected beta length 3, got %d", len(grouped["beta"]))
	}
	assertFloat(t, grouped, "beta", 0, 0.3)
	assertFloat(t, grouped, "beta", 1, 0.1)
	assertFloat(t, grouped, "beta", 2, -0.05)
}

func TestParseStanCSV_MultiRow(t *testing.T) {
	content := `# MCMC output
lp__,k,m,delta.1
-100.0,0.4,1.0,0.01
-99.5,0.6,1.1,0.02
-98.0,0.5,1.05,-0.01
`
	dir := t.TempDir()
	path := filepath.Join(dir, "output.csv")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	raw, err := ParseStanCSV(path)
	if err != nil {
		t.Fatalf("ParseStanCSV failed: %v", err)
	}

	if len(raw["k"]) != 3 {
		t.Fatalf("expected 3 samples for k, got %d", len(raw["k"]))
	}
	assertFloat(t, raw, "k", 0, 0.4)
	assertFloat(t, raw, "k", 1, 0.6)
	assertFloat(t, raw, "k", 2, 0.5)
}

func assertFloat(t *testing.T, m map[string][]float64, key string, idx int, expected float64) {
	t.Helper()
	vals, ok := m[key]
	if !ok {
		t.Fatalf("key %q not found", key)
	}
	if idx >= len(vals) {
		t.Fatalf("key %q: index %d out of range (len=%d)", key, idx, len(vals))
	}
	if diff := vals[idx] - expected; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("key %q[%d]: got %f, want %f", key, idx, vals[idx], expected)
	}
}
