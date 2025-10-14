#!/usr/bin/env python3
"""
Simple launcher for the split-view face emoji matcher.
"""

import sys
import os

def main():
    print("🎭 Face Emoji Matcher - Split View Launcher")
    print("=" * 50)
    print("Choose your view:")
    print("1. Simple Split View (face left, emoji right)")
    print("2. Enhanced View (with feature analysis)")
    print("3. Original View (overlay)")
    print()
    
    choice = input("Enter your choice (1-3, default 1): ").strip()
    
    if choice == "2":
        print("🚀 Launching Enhanced View...")
        os.system("python face_emoji_enhanced_view.py")
    elif choice == "3":
        print("🚀 Launching Original View...")
        os.system("python simple_demo_opencv.py")
    else:
        print("🚀 Launching Simple Split View...")
        os.system("python face_emoji_split_view.py")

if __name__ == "__main__":
    main()
