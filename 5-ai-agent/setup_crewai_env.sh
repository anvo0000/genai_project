#!/bin/bash

# Stop on error
set -e

echo "🔧 Installing dependencies for CrewAI setup..."

# Check for Homebrew and install pyenv if needed
if ! command -v brew &> /dev/null; then
  echo "❌ Homebrew not found. Please install Homebrew first."
  exit 1
fi

if ! command -v pyenv &> /dev/null; then
  echo "📦 Installing pyenv..."
  brew install pyenv
fi

if ! command -v pyenv-virtualenv &> /dev/null; then
  echo "📦 Installing pyenv-virtualenv..."
  brew install pyenv-virtualenv
fi

# Initialize pyenv in the shell
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.11 if not already
PYTHON_VERSION="3.11.9"
if ! pyenv versions | grep -q $PYTHON_VERSION; then
  echo "📥 Installing Python $PYTHON_VERSION..."
  pyenv install $PYTHON_VERSION
fi

# Create and activate virtualenv
ENV_NAME="crewai-env"
if ! pyenv virtualenvs | grep -q $ENV_NAME; then
  echo "📦 Creating virtualenv '$ENV_NAME'..."
  pyenv virtualenv $PYTHON_VERSION $ENV_NAME
fi

echo "🚀 Activating virtualenv '$ENV_NAME'..."
pyenv activate $ENV_NAME

# Upgrade pip and install requirements
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ CrewAI environment setup complete!"
echo "To activate the environment later, run: pyenv activate $ENV_NAME"
