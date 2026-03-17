package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/behnamebrahimi/neoprophet-go/internal/cmdstan"

	httpapi "github.com/behnamebrahimi/neoprophet-go/api/http"
)

func main() {
	httpAddr := flag.String("http", ":8080", "HTTP listen address")
	stanPath := flag.String("stan", "", "path to compiled CmdStan Prophet binary")
	flag.Parse()

	if *stanPath == "" {
		// Try environment variable
		*stanPath = os.Getenv("PROPHET_STAN_BINARY")
	}
	if *stanPath == "" {
		log.Fatal("must provide --stan flag or set PROPHET_STAN_BINARY env var")
	}

	if _, err := os.Stat(*stanPath); err != nil {
		log.Fatalf("Stan binary not found at %s: %v", *stanPath, err)
	}

	runner := cmdstan.NewRunner(*stanPath)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	log.Printf("Starting neoprophet-go server")
	log.Printf("Stan binary: %s", *stanPath)

	if err := httpapi.ListenAndServe(ctx, *httpAddr, runner); err != nil {
		log.Printf("server stopped: %v", err)
	}
}
