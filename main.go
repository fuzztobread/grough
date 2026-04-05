package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/dsp/fourier"
)

const (
	channels        = 1
	bitsPerSample   = 16
	maxFrequency    = 22050
	samplesPerSec   = maxFrequency * 2
	avgBytesPerSec  = channels * samplesPerSec * (bitsPerSample / 8)
	avgAmplitude    = 8.0
	durationSeconds = 20
)

func main() {
	kind := "white"
	if len(os.Args) > 1 {
		kind = os.Args[1]
	}

	var setup func(spectrum []complex128)

	switch kind {
	case "white", "":
		setup = func(spectrum []complex128) {
			for i := range spectrum {
				spectrum[i] = cmplx.Rect(avgAmplitude, rand.Float64()*2*math.Pi)
			}
		}
	case "pink":
		setup = func(spectrum []complex128) {
			normalization := avgAmplitude * math.Sqrt(float64(maxFrequency)/2.0)
			power := normalization * normalization
			for hz := 20; hz < len(spectrum); hz++ {
				amp := math.Sqrt(power / float64(hz+1))
				spectrum[hz] = cmplx.Rect(amp, rand.Float64()*2*math.Pi)
			}
		}
	case "brownian":
		setup = func(spectrum []complex128) {
			pinkNorm := avgAmplitude * math.Sqrt(float64(maxFrequency)/2.0) * 4.0
			pinkPower := pinkNorm * pinkNorm
			pinkMaxAmp := math.Sqrt(pinkPower / 20.0)
			for hz := 20; hz < len(spectrum); hz++ {
				amp := ((1.0 / float64(hz+1)) / (1.0 / float64(20+1))) * pinkMaxAmp
				spectrum[hz] = cmplx.Rect(amp, rand.Float64()*2*math.Pi)
			}
		}
	case "blue":
		setup = func(spectrum []complex128) {
			normalization := avgAmplitude / math.Sqrt(float64(maxFrequency)/2.0)
			power := normalization * normalization
			for hz := range spectrum {
				amp := math.Sqrt(power * float64(hz+1))
				spectrum[hz] = cmplx.Rect(amp, rand.Float64()*2*math.Pi)
			}
		}
	case "violet":
		setup = func(spectrum []complex128) {
			normalization := avgAmplitude / math.Sqrt(float64(maxFrequency)/2.0) / 4.0
			power := normalization * normalization
			for hz := range spectrum {
				amp := power * float64(hz+1)
				spectrum[hz] = cmplx.Rect(amp, rand.Float64()*2*math.Pi)
			}
		}
	case "grey":
		rA := func(hz float64) float64 {
			return (math.Pow(12194.0, 2) * math.Pow(hz, 4)) /
				((math.Pow(hz, 2) + math.Pow(20.6, 2)) *
					math.Sqrt((math.Pow(hz, 2)+math.Pow(107.7, 2))*(math.Pow(hz, 2)+math.Pow(737.9, 2))) *
					(math.Pow(hz, 2) + math.Pow(12194.0, 2)))
		}
		ra1000 := rA(1000.0)
		setup = func(spectrum []complex128) {
			for hz := 20; hz < len(spectrum); hz++ {
				aInDb := 20*math.Log10(rA(float64(hz))) - 20*math.Log10(ra1000)
				avgInDb := 20 * math.Log10(avgAmplitude)
				targetInDb := avgInDb - aInDb
				a := math.Pow(10.0, targetInDb/20.0)
				spectrum[hz] = cmplx.Rect(a, rand.Float64()*2*math.Pi)
			}
		}
	default:
		fmt.Fprintf(os.Stderr, "%s noise not supported (yet)\n", kind)
		os.Exit(1)
	}

	if err := generateNoise(setup); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func generateNoise(spectrumSetup func([]complex128)) error {
	sampleDataLen := uint32(avgBytesPerSec * durationSeconds)
	length := samplesPerSec

	f, err := os.Create("audio.wav")
	if err != nil {
		return err
	}
	defer f.Close()

	// RIFF header
	fmtChunkSize := 2 + 2 + 4 + 4 + 2 + 2 // fmt chunk fields: format_tag, channels, samples_per_sec, avg_bytes_per_sec, block_align, bits_per_sample
	riffSize := sampleDataLen + uint32(3*4+fmtChunkSize)

	write := func(v any) error {
		return binary.Write(f, binary.LittleEndian, v)
	}

	f.Write([]byte("RIFF"))
	write(riffSize)
	f.Write([]byte("WAVE"))

	// fmt chunk
	f.Write([]byte("fmt "))
	write(uint32(fmtChunkSize))
	write(uint16(0x0001))                          // PCM
	write(uint16(channels))                        // channels
	write(uint32(samplesPerSec))                   // samples per sec
	write(uint32(avgBytesPerSec))                  // avg bytes per sec
	write(uint16(channels * bitsPerSample / 8))    // block align
	write(uint16(bitsPerSample))                   // bits per sample

	// data chunk header
	f.Write([]byte("data"))
	write(sampleDataLen)

	// FFT setup — gonum uses real FFT; we simulate via full complex IFFT manually
	fft := fourier.NewCmplxFFT(length)

	spectrum := make([]complex128, length)
	half := length / 2

	dampen := -1.0

	for interval := 0; interval < durationSeconds; interval++ {
		if interval == 0 {
			spectrum[0] = 0
			spectrumSetup(spectrum[1:half])
		} else {
			for hz := 1; hz < half; hz++ {
				phase := rand.Float64() * (float64(hz) / 22050.0) * (math.Pi / 2.0)
				spectrum[hz] *= cmplx.Rect(1.0, phase)
			}
		}

		// populate conjugate mirror
		spectrum[0] = 0
		spectrum[half] = 0
		for i := 1; i < half; i++ {
			spectrum[length-i] = cmplx.Conj(spectrum[i])
		}

		// IFFT
		time := fft.Sequence(nil, spectrum)

		for _, sample := range time {
			re := real(sample)
			amplitude := math.Round(re)
			amplitude = amplitude + amplitude*dampen
			clamped := int16(clamp(int64(amplitude), math.MinInt16, math.MaxInt16))
			dampen = math.Min(dampen+0.0001, 0.0)
			write(clamped)
		}
	}

	return nil
}

func clamp(v, min, max int64) int64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
