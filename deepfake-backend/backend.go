package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
)

type Response struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

func main() {
	r := mux.NewRouter()

	// Endpoint to handle file upload
	r.HandleFunc("/upload", uploadFileHandler).Methods("POST")

	// Serve the static frontend HTML
	r.PathPrefix("/").Handler(http.StripPrefix("/", http.FileServer(http.Dir("./static/"))))

	// CORS support
	headersOk := handlers.AllowedHeaders([]string{"X-Requested-With", "Content-Type", "Authorization"})
	originsOk := handlers.AllowedOrigins([]string{"*"})
	methodsOk := handlers.AllowedMethods([]string{"GET", "HEAD", "POST", "OPTIONS"})

	// Start the server
	port := ":8080"
	fmt.Printf("Starting server on %s\n", port)
	log.Fatal(http.ListenAndServe(port, handlers.CORS(originsOk, headersOk, methodsOk)(r)))
}

// Handle file upload
func uploadFileHandler(w http.ResponseWriter, r *http.Request) {
	// Parse the uploaded file
	err := r.ParseMultipartForm(10 << 20) // Max file size: 10 MB
	if err != nil {
		http.Error(w, "Error parsing form", http.StatusBadRequest)
		return
	}

	// Get the file from the request
	file, handler, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Error retrieving file", http.StatusInternalServerError)
		return
	}
	defer file.Close()

	// Save the file locally
	filePath := filepath.Join("uploads", handler.Filename)
	f, err := os.Create(filePath)
	if err != nil {
		http.Error(w, "Error saving file", http.StatusInternalServerError)
		return
	}
	defer f.Close()
	io.Copy(f, file)

	// Run deepfake detection (assume a Python model is used)
	detectionResult, err := runDeepfakeDetection(filePath)
	if err != nil {
		http.Error(w, "Error processing video", http.StatusInternalServerError)
		return
	}

	// Return the response as JSON
	w.Header().Set("Content-Type", "application/json")
	response := Response{
		Success: true,
		Message: detectionResult,
	}
	json.NewEncoder(w).Encode(response)
}

// Function to run deepfake detection using Python
func runDeepfakeDetection(videoPath string) (string, error) {
	// Call the Python script for deepfake detection
	cmd := exec.Command("python", "python_script.py", videoPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("error running deepfake detection: %v", err)
	}

	// Extract the result ("Real" or "Fake") from output
	result := string(output)
	if result == "Real" || result == "Fake" {
		return result, nil
	}
	return "", fmt.Errorf("unexpected output from deepfake detection: %s", result)
}
