"""
Example usage scripts for the Luggage Comparison System

This file demonstrates various ways to use the LuggageComparator class
for different use cases and scenarios.
"""

import os
import argparse
from luggage_comparator import LuggageComparator


def compare_two_images_example():
    """Example: Compare two luggage images directly."""
    print("\n=== Two Image Comparison Example ===")
    
    comparator = LuggageComparator()
    
    # Replace with actual image paths
    image1 = "sample_images/luggage1.jpg"
    image2 = "sample_images/luggage2.jpg"
    
    if os.path.exists(image1) and os.path.exists(image2):
        try:
            similarity = comparator.compare_images(image1, image2)
            print(f"Similarity between images: {similarity:.2f}%")
            
            if similarity > 85:
                print("High similarity - likely the same luggage")
            elif similarity > 60:
                print("Medium similarity - possibly related luggage")
            else:
                print("Low similarity - likely different luggage")
                
        except Exception as e:
            print(f"Error comparing images: {e}")
    else:
        print("Sample images not found. Please provide valid image paths.")


def build_database_example():
    """Example: Build a searchable database of luggage images."""
    print("\n=== Database Building Example ===")
    
    comparator = LuggageComparator()
    
    # Sample luggage images directory
    images_dir = "sample_images/"
    
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Found {len(image_files)} images to add to database")
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(images_dir, image_file)
            image_id = f"luggage_{i:03d}_{image_file}"
            
            try:
                comparator.add_to_database(image_id, image_path)
                print(f"Added {image_id} to database")
            except Exception as e:
                print(f"Failed to add {image_file}: {e}")
        
        # Save database
        comparator.save_database("luggage_database.npz")
        print(f"Database saved with {len(comparator.embeddings_db)} images")
        
    else:
        print(f"Images directory '{images_dir}' not found")


def search_similar_example():
    """Example: Search for similar luggage in database."""
    print("\n=== Similar Image Search Example ===")
    
    comparator = LuggageComparator()
    
    # Load existing database
    if os.path.exists("luggage_database.npz"):
        comparator.load_database("luggage_database.npz")
        
        query_image = "query_luggage.jpg"
        
        if os.path.exists(query_image):
            try:
                similar_images = comparator.find_similar_images(
                    query_image, 
                    top_k=5, 
                    threshold=50.0
                )
                
                print(f"Found {len(similar_images)} similar images:")
                for image_id, similarity in similar_images:
                    print(f"  {image_id}: {similarity:.2f}% similar")
                    
            except Exception as e:
                print(f"Error searching for similar images: {e}")
        else:
            print(f"Query image '{query_image}' not found")
    else:
        print("No database found. Please run build_database_example first.")


def interactive_comparison():
    """Interactive mode for comparing luggage images."""
    print("\n=== Interactive Luggage Comparison ===")
    
    comparator = LuggageComparator()
    
    while True:
        print("\nOptions:")
        print("1. Compare two images")
        print("2. Add image to database")
        print("3. Search for similar images")
        print("4. Save database")
        print("5. Load database")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            img1 = input("Enter path to first image: ").strip()
            img2 = input("Enter path to second image: ").strip()
            
            if os.path.exists(img1) and os.path.exists(img2):
                try:
                    similarity = comparator.compare_images(img1, img2)
                    print(f"Similarity: {similarity:.2f}%")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("One or both images not found")
                
        elif choice == '2':
            img_path = input("Enter image path: ").strip()
            img_id = input("Enter image ID: ").strip()
            
            if os.path.exists(img_path):
                try:
                    comparator.add_to_database(img_id, img_path)
                    print(f"Added {img_id} to database")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Image not found")
                
        elif choice == '3':
            query_img = input("Enter query image path: ").strip()
            
            if os.path.exists(query_img):
                try:
                    top_k = int(input("Number of results (default 5): ") or "5")
                    threshold = float(input("Similarity threshold % (default 0): ") or "0")
                    
                    results = comparator.find_similar_images(query_img, top_k, threshold)
                    
                    if results:
                        print(f"Found {len(results)} similar images:")
                        for img_id, sim in results:
                            print(f"  {img_id}: {sim:.2f}%")
                    else:
                        print("No similar images found")
                        
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Query image not found")
                
        elif choice == '4':
            filepath = input("Enter save path (default: luggage_db.npz): ") or "luggage_db.npz"
            comparator.save_database(filepath)
            
        elif choice == '5':
            filepath = input("Enter database path: ").strip()
            if os.path.exists(filepath):
                comparator.load_database(filepath)
            else:
                print("Database file not found")
                
        elif choice == '6':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice")


def batch_process_directory():
    """Process all images in a directory and generate similarity matrix."""
    print("\n=== Batch Directory Processing ===")
    
    directory = input("Enter directory path: ").strip()
    
    if not os.path.exists(directory):
        print("Directory not found")
        return
    
    comparator = LuggageComparator()
    
    # Get all image files
    image_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No image files found in directory")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    # Process all images and build embeddings
    embeddings = {}
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(directory, img_file)
        try:
            embedding = comparator.process_image(img_path)
            embeddings[img_file] = embedding
            print(f"Processed {i+1}/{len(image_files)}: {img_file}")
        except Exception as e:
            print(f"Failed to process {img_file}: {e}")
    
    # Generate similarity matrix
    print("\nGenerating similarity matrix...")
    processed_files = list(embeddings.keys())
    
    print("\nSimilarity Matrix (%):")
    print("File\t" + "\t".join([f[:10] for f in processed_files]))
    
    for i, file1 in enumerate(processed_files):
        row = [file1[:10]]
        for j, file2 in enumerate(processed_files):
            if i == j:
                similarity = 100.0
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity([embeddings[file1]], [embeddings[file2]])[0][0]
                similarity = (sim + 1) / 2 * 100
            row.append(f"{similarity:.1f}")
        print("\t".join(row))


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Luggage Comparison System Examples")
    parser.add_argument(
        "--mode", 
        choices=["compare", "database", "search", "interactive", "batch"],
        default="interactive",
        help="Mode to run"
    )
    
    args = parser.parse_args()
    
    print("Luggage Comparison System - Examples")
    print("===================================")
    
    if args.mode == "compare":
        compare_two_images_example()
    elif args.mode == "database":
        build_database_example()
    elif args.mode == "search":
        search_similar_example()
    elif args.mode == "interactive":
        interactive_comparison()
    elif args.mode == "batch":
        batch_process_directory()
    
    print("\nDone!")


if __name__ == "__main__":
    main()