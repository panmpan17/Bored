package main

import (
	"fmt"
	// "flag"
	"github.com/spf13/cobra"
)

func main() {
	// No package at all
	// if len(os.Args) < 2 {
	// 	fmt.Println("Usage: go run test_cli.go <command>")
	// 	return
	// }

	// command := os.Args[1]

	// switch command {
	// 	case "greet":
	// 		fmt.Println("Hello! How can I assist you today?")
	// 	case "exit":
	// 		fmt.Println("Goodbye! Have a great day!")
	// 	default:
	// 		fmt.Println("Unknown command. Available commands: greet, exit")
	// 		fmt.Println("Type 'greet' to receive a greeting or 'exit' to quit the program.")
	// }

	// Using flag package to parse command line arguments
	// author := flag.String("author", "Unknown", "Author of the program")
	// version := flag.String("version", "1.0.0", "Version of the program")
	// help := flag.Bool("help", false, "Display help information")
	// flag.Parse()

	// fmt.Printf("Author: %s\n", *author)
	// fmt.Printf("Version: %s\n", *version)
	// fmt.Printf("help: %s\n", *help)

	var author string

	var rootCmd = &cobra.Command{
		Use:   "notecli",
		Short: "notecli is a simple CLI to post messages",
	}

	var postCmd = &cobra.Command{
		Use:   "post",
		Short: "Post a new message",
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) == 0 {
				fmt.Println("Please provide a message.")
				return
			}
			message := args[0]
			fmt.Printf("Posted by %s: %s\n", author, message)
		},
	}

	postCmd.Flags().StringVarP(&author, "author", "a", "Anonymous", "Author of the message")
	rootCmd.AddCommand(postCmd)

	rootCmd.Execute()
	// go run test_cli.go post "What's up" --author Michael
}