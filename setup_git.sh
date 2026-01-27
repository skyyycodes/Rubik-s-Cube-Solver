#!/bin/bash
# Setup script to initialize a fresh git repository

cd "$(dirname "$0")"

echo "Removing old .git directory..."
rm -rf .git

echo "Initializing new git repository..."
git init --initial-branch=main

echo "Adding all files..."
git add .

echo "Creating initial commit..."
git commit -m "Initial commit: Rubik's Cube Solver with 3D visualization

Features:
- 3D interactive cube input interface
- Step-by-step solution visualization with real-time cube updates
- Clear English instructions for each move
- Support for manual and camera-based input modes
- Kociemba algorithm integration for optimal solutions"

echo ""
echo "Repository initialized successfully!"
echo "Run 'git log' to see the commit history"
echo "Run 'git status' to see current status"
