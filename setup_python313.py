#!/usr/bin/env python3
"""
Python 3.13 compatible setup script for Face Emoji Matcher.
This script handles the distutils removal issue in Python 3.13.
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

def install_dependencies_python313():
    """Install dependencies with Python 3.13 compatibility fixes."""
    print("📦 Installing dependencies for Python 3.13...")
    
    pip_cmd = get_pip_command()
    
    # First, upgrade pip and install setuptools
    print("🔄 Upgrading pip and installing setuptools...")
    if not run_command(f"{pip_cmd} install --upgrade pip setuptools wheel", "Upgrading pip and setuptools"):
        return False
    
    # Install packages one by one to handle compatibility issues
    packages = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0", 
        "pillow>=10.0.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "mediapipe>=0.10.0"
    ]
    
    for package in packages:
        print(f"🔄 Installing {package}...")
        if not run_command(f"{pip_cmd} install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, trying alternative approach...")
            # Try with --no-build-isolation for problematic packages
            if not run_command(f"{pip_cmd} install --no-build-isolation {package}", f"Installing {package} with --no-build-isolation"):
                print(f"❌ Could not install {package}")
                return False
    
    return True

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
    print("🎭 Face Emoji Matcher Setup (Python 3.13 Compatible)")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Install dependencies with Python 3.13 fixes
    if not install_dependencies_python313():
        print("\n⚠️  Some packages failed to install.")
        print("   This might be due to Python 3.13 compatibility issues.")
        print("   You can try:")
        print("   1. Use Python 3.11 or 3.12 instead")
        print("   2. Install packages manually in the virtual environment")
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but basic setup is complete")
        print("   You can still try running the application")
    
    print("\n" + "=" * 50)
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
