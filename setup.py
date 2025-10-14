#!/usr/bin/env python3
"""
Setup script for Face Emoji Matcher.
This script helps install dependencies and verify the installation.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("   Please install Python 3.7 or higher.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    print("🔧 Setting up virtual environment...")
    
    if os.path.exists("venv"):
        print("✅ Virtual environment 'venv' already exists")
        return True
    
    if run_command("python3 -m venv venv", "Creating virtual environment"):
        print("✅ Virtual environment created successfully")
        return True
    else:
        print("❌ Failed to create virtual environment")
        return False

def get_pip_command():
    """Get the correct pip command for the virtual environment."""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        return "venv/bin/pip"

def install_dependencies():
    """Install required dependencies in virtual environment."""
    print("📦 Installing dependencies in virtual environment...")
    
    pip_cmd = get_pip_command()
    
    if run_command(f"{pip_cmd} install -r requirements.txt", "Installing packages in virtual environment"):
        return True
    
    print("❌ Failed to install dependencies")
    print("   Please try installing manually:")
    print(f"   {pip_cmd} install opencv-python numpy mediapipe pillow flask flask-cors")
    return False

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    required_modules = [
        'cv2',
        'numpy',
        'mediapipe',
        'PIL',
        'flask',
        'flask_cors'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All modules imported successfully")
        return True

def get_python_command():
    """Get the correct python command for the virtual environment."""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        return "venv/bin/python"

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    python_cmd = get_python_command()
    
    if run_command(f"{python_cmd} test_system.py", "Running system tests"):
        return True
    else:
        print("❌ Tests failed to run")
        return False

def main():
    """Main setup function."""
    print("🎭 Face Emoji Matcher Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but basic setup is complete")
        print("   You can still try running the application")
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed!")
    print("\nTo activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("  venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("  source venv/bin/activate")
    print("\nTo run the application:")
    print("  Web interface: python app.py")
    print("  Command line:  python simple_demo.py")
    print("\nThen open your browser to: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
