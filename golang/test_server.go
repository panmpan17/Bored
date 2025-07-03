package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
)

// Message represents a single post on the billboard
type Message struct {
	Author  string `json:"author"`
	Content string `json:"content"`
}

// Global in-memory message list with mutex for safe concurrent access
var (
	messages []Message
	mutex    sync.Mutex
)

func main() {
	http.Handle("/", http.FileServer(http.Dir("./static")))

	http.HandleFunc("/messages", messagesHandler)
	log.Println("Server started at http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

// messagesHandler handles both GET and POST to /messages
func messagesHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
		case "GET":
			getMessages(w, r)
		case "POST":
			postMessage(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// getMessages returns the list of messages as JSON
func getMessages(w http.ResponseWriter, r *http.Request) {
	mutex.Lock()
	defer mutex.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(messages)
}

// postMessage adds a new message from the request body
func postMessage(w http.ResponseWriter, r *http.Request) {
	var msg Message
	err := json.NewDecoder(r.Body).Decode(&msg)
	if err != nil || msg.Author == "" || msg.Content == "" {
		http.Error(w, "Invalid message data", http.StatusBadRequest)
		return
	}

	mutex.Lock()
	messages = append(messages, msg)
	mutex.Unlock()

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(msg)
}
