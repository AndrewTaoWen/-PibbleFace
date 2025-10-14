#!/usr/bin/env python3
"""
Test script for the OpenCV version of Face Emoji Matcher.
This script tests the core functionality without requiring a camera.
"""

import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

def test_emoji_database():
    """Test that the emoji database is properly loaded."""
    print("🧪 Testing emoji database...")
    matcher = FaceEmojiMatcherOpenCV()
    
    # Check if database is loaded
    assert len(matcher.emoji_database) > 0, "Emoji database is empty"
    print(f"✅ Emoji database loaded with {len(matcher.emoji_database)} emojis")
    
    # Check if all emojis have required features
    required_features = ['smile_intensity', 'eye_openness', 'eyebrow_raise', 'mouth_width']
    for emoji, features in matcher.emoji_database.items():
        for feature in required_features:
            assert feature in features, f"Emoji {emoji} missing feature {feature}"
            assert 0 <= features[feature] <= 1, f"Emoji {emoji} feature {feature} out of range"
    
    print("✅ All emojis have valid features")
    return True

def test_similarity_calculation():
    """Test the similarity calculation function."""
    print("🧪 Testing similarity calculation...")
    matcher = FaceEmojiMatcherOpenCV()
    
    # Test with identical features
    face_features = {
        'smile_intensity': 0.5,
        'eye_openness': 0.8,
        'eyebrow_raise': 0.3,
        'mouth_width': 0.6
    }
    
    emoji_features = {
        'smile_intensity': 0.5,
        'eye_openness': 0.8,
        'eyebrow_raise': 0.3,
        'mouth_width': 0.6
    }
    
    similarity = matcher.calculate_similarity(face_features, emoji_features)
    assert similarity == 1.0, f"Identical features should have similarity 1.0, got {similarity}"
    print("✅ Identical features similarity test passed")
    
    # Test with completely different features
    emoji_features_different = {
        'smile_intensity': 0.0,
        'eye_openness': 0.0,
        'eyebrow_raise': 0.0,
        'mouth_width': 0.0
    }
    
    similarity = matcher.calculate_similarity(face_features, emoji_features_different)
    assert similarity < 0.5, f"Very different features should have low similarity, got {similarity}"
    print("✅ Different features similarity test passed")
    
    # Test with None face features
    similarity = matcher.calculate_similarity(None, emoji_features)
    assert similarity == 0.0, f"None face features should have similarity 0.0, got {similarity}"
    print("✅ None face features test passed")
    
    return True

def test_emoji_matching():
    """Test the emoji matching function."""
    print("🧪 Testing emoji matching...")
    matcher = FaceEmojiMatcherOpenCV()
    
    # Test with a happy face
    happy_features = {
        'smile_intensity': 0.8,
        'eye_openness': 0.9,
        'eyebrow_raise': 0.2,
        'mouth_width': 0.7
    }
    
    best_emoji, similarity = matcher.find_best_emoji_match(happy_features)
    assert best_emoji in matcher.emoji_database, f"Best emoji {best_emoji} not in database"
    assert 0 <= similarity <= 1, f"Similarity {similarity} out of range"
    print(f"✅ Happy face matched to {best_emoji} with similarity {similarity:.2f}")
    
    # Test with a sad face
    sad_features = {
        'smile_intensity': 0.0,
        'eye_openness': 0.6,
        'eyebrow_raise': 0.6,
        'mouth_width': 0.3
    }
    
    best_emoji, similarity = matcher.find_best_emoji_match(sad_features)
    assert best_emoji in matcher.emoji_database, f"Best emoji {best_emoji} not in database"
    assert 0 <= similarity <= 1, f"Similarity {similarity} out of range"
    print(f"✅ Sad face matched to {best_emoji} with similarity {similarity:.2f}")
    
    # Test with None features
    best_emoji, similarity = matcher.find_best_emoji_match(None)
    assert best_emoji == '😐', f"None features should default to 😐, got {best_emoji}"
    assert similarity == 0.0, f"None features should have similarity 0.0, got {similarity}"
    print("✅ None features test passed")
    
    return True

def test_face_feature_extraction():
    """Test face feature extraction with a dummy image."""
    print("🧪 Testing face feature extraction...")
    matcher = FaceEmojiMatcherOpenCV()
    
    # Create a dummy image (black image)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # This should return None since there's no face
    features = matcher.extract_facial_features(dummy_image)
    assert features is None, f"Empty image should return None features, got {features}"
    print("✅ Empty image test passed")
    
    return True

def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Face Emoji Matcher Tests (OpenCV Version)")
    print("=" * 60)
    
    try:
        test_emoji_database()
        test_similarity_calculation()
        test_emoji_matching()
        test_face_feature_extraction()
        
        print("=" * 60)
        print("🎉 All tests passed! The OpenCV version is working correctly.")
        print("You can now run the application with:")
        print("  - Command line: python simple_demo_opencv.py")
        print("  - Note: Web interface requires MediaPipe (not available for Python 3.13)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_all_tests()
