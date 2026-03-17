package cmdstan

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ParseStanCSV reads a CmdStan CSV output file and returns parameter values.
// Comment lines (starting with #) are skipped. The first non-comment line is
// the header. Remaining lines are data rows. Indexed parameters like delta.1,
// delta.2 are grouped into a single "delta" key.
func ParseStanCSV(path string) (map[string][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	var headerLine string
	var dataLines []string

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
			continue
		}
		if headerLine == "" {
			headerLine = line
		} else {
			dataLines = append(dataLines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanning %s: %w", path, err)
	}

	if headerLine == "" {
		return nil, fmt.Errorf("no header found in %s", path)
	}
	if len(dataLines) == 0 {
		return nil, fmt.Errorf("no data rows found in %s", path)
	}

	headers := strings.Split(headerLine, ",")
	raw := make(map[string][]float64, len(headers))

	for _, line := range dataLines {
		fields := strings.Split(line, ",")
		if len(fields) != len(headers) {
			return nil, fmt.Errorf("row has %d fields, expected %d", len(fields), len(headers))
		}
		for i, h := range headers {
			val, err := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
			if err != nil {
				return nil, fmt.Errorf("parsing field %q value %q: %w", h, fields[i], err)
			}
			raw[h] = append(raw[h], val)
		}
	}

	return raw, nil
}

// GroupParams groups indexed Stan parameter columns (e.g., delta.1, delta.2)
// into a single key ("delta") with concatenated values per sample.
// Scalar parameters are left unchanged.
func GroupParams(raw map[string][]float64, nSamples int) map[string][]float64 {
	// First pass: discover base names and their indices
	type indexedParam struct {
		baseName string
		index    int
	}

	baseNames := make(map[string]int) // baseName -> max index seen
	indexed := make(map[string]bool)

	for key := range raw {
		parts := strings.SplitN(key, ".", 2)
		if len(parts) == 2 {
			if idx, err := strconv.Atoi(parts[1]); err == nil {
				base := parts[0]
				indexed[base] = true
				if idx > baseNames[base] {
					baseNames[base] = idx
				}
				continue
			}
		}
		// Scalar or non-indexed
		if _, exists := baseNames[key]; !exists {
			baseNames[key] = 0
		}
	}

	result := make(map[string][]float64)

	for base, maxIdx := range baseNames {
		if !indexed[base] {
			// Scalar parameter: copy directly
			result[base] = raw[base]
			continue
		}

		// Indexed parameter: interleave per sample
		// For nSamples=1, result is [v1, v2, ..., v_maxIdx]
		// For nSamples>1, result is [s1_v1, s1_v2, ..., s1_vN, s2_v1, ...]
		paramSize := maxIdx
		grouped := make([]float64, 0, nSamples*paramSize)

		for s := 0; s < nSamples; s++ {
			for i := 1; i <= maxIdx; i++ {
				key := fmt.Sprintf("%s.%d", base, i)
				vals := raw[key]
				if s < len(vals) {
					grouped = append(grouped, vals[s])
				}
			}
		}
		result[base] = grouped
	}

	return result
}
