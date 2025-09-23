#!/usr/bin/env python3
"""
Test script for the genetic algorithm with pygame visualization
"""

import sys
import os
import pygame

# Add the genetic directory to the path so we can import the modules
sys.path.append(os.path.dirname(__file__))

from snake_game_genetic import test_genetic_individual

def main():
    print("🐍 Genetic Algorithm Snake AI Test with Visualization")
    print("=" * 60)

    # Look for trained models
    model_files = [
        '../genetic_snake_final.pth',
        '../genetic_snake_gen_870.pth',
        '../genetic_snake_gen_860.pth',
        '../genetic_snake_gen_850.pth',
        '../genetic_snake_gen_840.pth',
        '../genetic_snake_gen_830.pth',
        '../genetic_snake_gen_820.pth',
        '../genetic_snake_gen_810.pth',
        '../genetic_snake_gen_800.pth',
    ]

    # Find the best available model
    model_to_test = None
    for model_file in model_files:
        if os.path.exists(model_file):
            model_to_test = model_file
            break

    if model_to_test is None:
        print("❌ No trained genetic algorithm models found!")
        print("Available models should be in the parent directory:")
        for model_file in model_files:
            print(f"   - {model_file}")
        print("\nPlease train a model first using train_genetic.py")
        return

    print(f"🎯 Found model: {model_to_test}")
    print(f"🎮 Starting visualization test...")
    print("📝 Controls: Close the pygame window to stop testing")
    print("=" * 60)

    try:
        # Test with visualization
        scores = test_genetic_individual(
            model_path=model_to_test,
            num_games=5,
            display=True
        )

        if scores:
            print("\n🎉 Test completed successfully!")
            print(f"🏆 Best score in this test: {max(scores)}")
            print(f"📊 Average score: {sum(scores)/len(scores):.2f}")
        else:
            print("\n⚠️  Test was interrupted or no games completed")

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up pygame
        try:
            pygame.quit()
        except:
            pass

if __name__ == "__main__":
    main()
