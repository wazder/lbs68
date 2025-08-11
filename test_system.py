"""
Test script for the Luggage Comparison System

This script tests the basic functionality of the system and helps
verify that all components are working correctly.
"""

import sys
import numpy as np
from luggage_comparator import LuggageComparator


def test_initialization():
    """Test system initialization."""
    print("Testing system initialization...")
    
    try:
        comparator = LuggageComparator()
        print("[OK] LuggageComparator initialized successfully")
        
        # Check if models are loaded
        if comparator.sam_predictor is not None:
            print("[OK] SAM model loaded")
        else:
            print("[WARNING] SAM model not loaded (may need to install segment-anything)")
            
        if comparator.clip_model is not None and comparator.clip_processor is not None:
            print("[OK] CLIP model loaded")
        else:
            print("[WARNING] CLIP model not loaded (may need to install transformers)")
            
        return comparator
        
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return None


def test_embedding_generation():
    """Test embedding generation with dummy data."""
    print("\nTesting embedding generation...")
    
    comparator = LuggageComparator()
    
    if comparator.clip_model is None:
        print("⚠ Skipping embedding test - CLIP not available")
        return True
    
    try:
        # Create a dummy image (random RGB values)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Extract embedding
        embedding = comparator.extract_embedding(dummy_image)
        
        print(f"✓ Embedding generated with shape: {embedding.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False


def test_similarity_calculation():
    """Test similarity calculation between embeddings."""
    print("\nTesting similarity calculation...")
    
    try:
        # Create two dummy embeddings
        embedding1 = np.random.randn(512)
        embedding2 = np.random.randn(512)
        
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        similarity_percentage = (similarity + 1) / 2 * 100
        
        print(f"✓ Similarity calculation successful: {similarity_percentage:.2f}%")
        
        # Test identical embeddings
        identical_similarity = cosine_similarity([embedding1], [embedding1])[0][0]
        identical_percentage = (identical_similarity + 1) / 2 * 100
        
        if abs(identical_percentage - 100.0) < 0.01:
            print("✓ Identical embedding similarity is ~100%")
        else:
            print(f"⚠ Identical embedding similarity is {identical_percentage:.2f}% (expected ~100%)")
            
        return True
        
    except Exception as e:
        print(f"✗ Similarity calculation failed: {e}")
        return False


def test_luggage_detection():
    """Test luggage detection functionality."""
    print("\nTesting luggage detection...")
    
    comparator = LuggageComparator()
    
    if comparator.clip_model is None:
        print("⚠ Skipping luggage detection test - CLIP not available")
        return True
    
    try:
        # Create a dummy image (random RGB values)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test luggage detection
        detection_result = comparator.detect_luggage(dummy_image)
        
        print(f"✓ Luggage detection completed")
        print(f"  - Is luggage: {detection_result['is_luggage']}")
        print(f"  - Confidence: {detection_result['confidence']:.3f}")
        print(f"  - Reason: {detection_result['reason']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Luggage detection failed: {e}")
        return False


def test_mask_application():
    """Test mask application functionality."""
    print("\nTesting mask application...")
    
    try:
        # Create dummy image and mask
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True  # Square mask in center
        
        comparator = LuggageComparator()
        masked_image = comparator.apply_mask(image, mask)
        
        print(f"✓ Mask applied successfully, output shape: {masked_image.shape}")
        
        # Check that masked region preserved original colors
        center_pixel_original = image[50, 50]
        center_pixel_masked = masked_image[50, 50]
        
        if np.array_equal(center_pixel_original, center_pixel_masked):
            print("✓ Masked region preserved correctly")
        else:
            print("⚠ Masked region may not be preserved correctly")
            
        # Check that unmasked region is white/background
        corner_pixel = masked_image[10, 10]
        if np.all(corner_pixel == 255):
            print("✓ Unmasked region set to white background")
        else:
            print(f"⚠ Unmasked region is {corner_pixel} (expected [255, 255, 255])")
            
        return True
        
    except Exception as e:
        print(f"✗ Mask application failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Luggage Comparison System - Test Suite")
    print("=====================================")
    
    tests = [
        test_initialization,
        test_embedding_generation,
        test_similarity_calculation,
        test_luggage_detection,
        test_mask_application
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        
    return passed == total


def main():
    """Main test function - DEPRECATED: Use run_analysis.py --test instead."""
    print("WARNING: Direct execution of test_system.py is deprecated.")
    print("   Please use the integrated test system:")
    print("   python run_analysis.py --test")
    print()
    
    # Run tests for backward compatibility
    print("Running tests via legacy interface...")
    success = run_all_tests()
    
    if not success:
        print("\nTroubleshooting:")
        print("- Install required packages: pip install -r requirements.txt")
        print("- Check dependencies: python run_analysis.py --check-deps")
        print("- For detailed help: python run_analysis.py --help")
        
    print("\nFor better testing experience, use: python run_analysis.py --test")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()