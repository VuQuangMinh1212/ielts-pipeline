"""
Process audio/video files to create training dataset
"""
import os
import json
from pathlib import Path
from typing import Dict, List


def process_audio_file(
    audio_path: str,
    output_dir: str = "./data/processed"
) -> Dict:
    """
    Process single audio file
    For now, creates a sample entry - you can integrate with Gemini API later
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filename = Path(audio_path).stem
    
    # Sample data - in production, use Gemini API to transcribe
    sample_data = {
        "file": filename,
        "transcript": "Sample transcript - integrate with Gemini API for real transcription",
        "scores": {
            "fluency": 6.5,
            "lexical_resource": 6.0,
            "grammatical_range": 6.5,
            "pronunciation": 6.5,
        },
        "metadata": {
            "source": audio_path,
            "band": "6.5",
        }
    }
    
    # Save processed data
    output_file = Path(output_dir) / f"{filename}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Processed: {filename}")
    print(f"  Output: {output_file}")
    
    return sample_data


def process_all_audio_files(
    input_dir: str = "./data training",
    output_dir: str = "./data/processed"
):
    """Process all audio files in directory"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"✗ Directory not found: {input_dir}")
        return []
        
    # Supported audio/video formats
    audio_extensions = {'.mp3', '.mp4', '.wav', '.m4a', '.ogg', '.webm'}
    
    processed = []
    
    for file in input_path.iterdir():
        if file.suffix.lower() in audio_extensions:
            try:
                data = process_audio_file(str(file), output_dir)
                processed.append(data)
            except Exception as e:
                print(f"✗ Error processing {file}: {e}")
                
    print(f"\n✓ Processed {len(processed)} files")
    return processed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files for training")
    parser.add_argument(
        "--input",
        type=str,
        default="./data training",
        help="Input directory with audio files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    
    args = parser.parse_args()
    
    process_all_audio_files(args.input, args.output)
