package prophet

import (
	"math"
	"testing"
)

func TestFourierFeatures(t *testing.T) {
	// Python formula: sin((j/period) * 2*pi*t), cos((j/period) * 2*pi*t)
	// At t=0: sin(0)=0, cos(0)=1 for all orders
	tvals := []float64{0, 182.625, 365.25}
	period := 365.25
	order := 2
	features := FourierFeatures(tvals, period, order)

	cols := 2 * order
	if len(features) != len(tvals)*cols {
		t.Fatalf("expected %d elements, got %d", len(tvals)*cols, len(features))
	}

	// t=0: sin(0)=0, cos(0)=1 for all orders
	assertNear(t, features[0], 0.0, "t=0, sin(1)")
	assertNear(t, features[1], 1.0, "t=0, cos(1)")
	assertNear(t, features[2], 0.0, "t=0, sin(2)")
	assertNear(t, features[3], 1.0, "t=0, cos(2)")

	// t=365.25 (one full period): sin(2pi)=0, cos(2pi)=1
	row2 := features[2*cols:]
	assertNear(t, row2[0], 0.0, "t=365.25, sin(1)")
	assertNear(t, row2[1], 1.0, "t=365.25, cos(1)")

	// t=182.625 (half period): sin(pi)=0, cos(pi)=-1
	row1 := features[cols:]
	assertNear(t, row1[0], 0.0, "t=182.625, sin(1)")
	assertNear(t, row1[1], -1.0, "t=182.625, cos(1)")
}

func TestFourierFeatures_MatchesPython(t *testing.T) {
	// Verify against Python Prophet's fourier_series output
	// Python: t = days since epoch, c = (j/period) * 2*pi*t
	// sin(c), cos(c)
	tDays := []float64{18262.0} // 2020-01-01 in days since epoch
	period := 365.25
	order := 1

	features := FourierFeatures(tDays, period, order)

	// Expected: sin((1/365.25) * 2*pi*18262), cos((1/365.25) * 2*pi*18262)
	c := 1.0 / period * 2.0 * math.Pi * 18262.0
	expectedSin := math.Sin(c)
	expectedCos := math.Cos(c)

	assertNear(t, features[0], expectedSin, "sin match")
	assertNear(t, features[1], expectedCos, "cos match")
}

func TestMakeSeasonalityFeatures(t *testing.T) {
	tDays := []float64{0, 1, 2, 3}
	specs := []SeasonalitySpec{
		{Name: "yearly", Period: 365.25, FourierOrder: 2, PriorScale: 10.0, Mode: ModeAdditive},
		{Name: "weekly", Period: 7.0, FourierOrder: 1, PriorScale: 10.0, Mode: ModeAdditive},
	}

	X, K, sA, sM, sigmas := MakeSeasonalityFeatures(tDays, specs)

	expectedK := 2*2 + 2*1 // 4 yearly + 2 weekly = 6
	if K != expectedK {
		t.Fatalf("expected K=%d, got %d", expectedK, K)
	}
	if len(X) != len(tDays)*K {
		t.Fatalf("expected X length %d, got %d", len(tDays)*K, len(X))
	}

	for i, v := range sA {
		if v != 1.0 {
			t.Errorf("sA[%d] = %f, want 1.0", i, v)
		}
	}
	for i, v := range sM {
		if v != 0.0 {
			t.Errorf("sM[%d] = %f, want 0.0", i, v)
		}
	}
	for i, v := range sigmas {
		if v != 10.0 {
			t.Errorf("sigmas[%d] = %f, want 10.0", i, v)
		}
	}
}

func assertNear(t *testing.T, got, want float64, label string) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s: got %f, want %f", label, got, want)
	}
}
