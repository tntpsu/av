#!/bin/bash
# Test script to show what launch_unity.sh would do without actually launching
source ./launch_unity.sh 2>&1 | grep -E "(Found Unity|Project path|Command:|Mode:)" || {
    # If sourcing doesn't work, let's manually test the logic
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    UNITY_PROJECT_PATH="$SCRIPT_DIR/unity/AVSimulation"
    
    echo "Testing Unity detection..."
    if [ -d "/Applications/Unity Hub.app" ]; then
        for version_dir in /Applications/Unity/Hub/Editor/*/; do
            if [ -d "$version_dir" ] && [ -f "${version_dir}Unity.app/Contents/MacOS/Unity" ]; then
                UNITY_PATH="${version_dir}Unity.app/Contents/MacOS/Unity"
                echo "✓ Found Unity: $UNITY_PATH"
                echo "✓ Project path: $UNITY_PROJECT_PATH"
                echo "✓ Command would be: $UNITY_PATH -projectPath \"$UNITY_PROJECT_PATH\""
                exit 0
            fi
        done
    fi
    echo "✗ Unity not found"
    exit 1
}
