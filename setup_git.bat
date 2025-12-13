@echo off
REM Setup script for pushing to GitHub (Windows)
REM Run this after creating a repository on GitHub

echo Initializing git repository...
git init

echo Adding all files...
git add .

echo Creating initial commit...
git commit -m "Initial commit: Super Mario Agent with Rainbow DQN"

echo.
echo Git repository initialized and initial commit created!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub (don't initialize with README)
echo 2. Run these commands (replace YOUR_USERNAME and YOUR_REPO_NAME):
git remote add origin https://github.com/pronzzz/super-mario-agent.git
git branch -M main
git push -u origin main
