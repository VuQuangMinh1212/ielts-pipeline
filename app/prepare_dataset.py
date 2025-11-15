"""
Script to prepare IELTS training dataset from audio transcriptions
"""
import json
import os
from pathlib import Path
from typing import List, Dict


def create_training_sample(transcript: str, scores: Dict[str, float]) -> Dict:
    """Create a single training sample"""
    return {
        "transcript": transcript,
        "scores": {
            "fluency": scores.get("fluency", 0.0),
            "lexical_resource": scores.get("lexical_resource", 0.0),
            "grammatical_range": scores.get("grammatical_range", 0.0),
            "pronunciation": scores.get("pronunciation", 0.0),
        }
    }


def prepare_dataset(
    input_dir: str = "./data/processed",
    output_file: str = "./data/ielts_training_data.jsonl"
):
    """
    Prepare training dataset from processed IELTS recordings
    
    Expected input format (JSON files):
    {
        "transcript": "...",
        "scores": {
            "fluency": 7.0,
            "lexical_resource": 6.5,
            ...
        }
    }
    """
    input_path = Path(input_dir)
    samples = []
    
    # Process all JSON files in input directory
    if input_path.exists():
        for json_file in input_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                if "transcript" in data and "scores" in data:
                    sample = create_training_sample(
                        data["transcript"],
                        data["scores"]
                    )
                    samples.append(sample)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                
    # If no samples found, create example dataset
    if not samples:
        print("No processed data found. Creating example dataset...")
        samples = create_example_dataset()
        
    # Write to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print(f"✓ Created dataset with {len(samples)} samples")
    print(f"✓ Saved to {output_file}")
    
    return samples


def create_example_dataset() -> List[Dict]:
    """Create example IELTS training dataset"""
    examples = [
        {
            "transcript": "Well, I think that technology has greatly influenced our daily lives. For example, smartphones allow us to stay connected with friends and family. Moreover, the internet provides us with access to vast amounts of information.",
            "scores": {
                "fluency": 7.5,
                "lexical_resource": 7.0,
                "grammatical_range": 7.5,
                "pronunciation": 7.0,
            }
        },
        {
            "transcript": "Uh, I like to, um, spend time with my family on weekends. We usually go to park or sometimes watch movie. It's very nice and relaxing.",
            "scores": {
                "fluency": 5.5,
                "lexical_resource": 5.0,
                "grammatical_range": 5.0,
                "pronunciation": 6.0,
            }
        },
        {
            "transcript": "In my opinion, environmental protection is extremely important. We need to reduce our carbon footprint and adopt sustainable practices. This includes recycling, using renewable energy, and minimizing waste.",
            "scores": {
                "fluency": 8.0,
                "lexical_resource": 8.5,
                "grammatical_range": 8.0,
                "pronunciation": 7.5,
            }
        },
        {
            "transcript": "I not very good at speaking English. Sometimes I making mistake with grammar. But I try my best to improve every day.",
            "scores": {
                "fluency": 4.5,
                "lexical_resource": 4.0,
                "grammatical_range": 4.0,
                "pronunciation": 5.5,
            }
        },
        {
            "transcript": "Travel is one of my greatest passions. I've been fortunate enough to visit numerous countries across different continents. Each destination offers unique cultural experiences and perspectives that broaden one's understanding of the world.",
            "scores": {
                "fluency": 8.5,
                "lexical_resource": 9.0,
                "grammatical_range": 8.5,
                "pronunciation": 8.0,
            }
        },
    ]
    
    return examples


def validate_dataset(dataset_file: str = "./data/ielts_training_data.jsonl"):
    """Validate the training dataset"""
    with open(dataset_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
        
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(samples)}")
    
    # Calculate average scores
    avg_scores = {
        "fluency": 0,
        "lexical_resource": 0,
        "grammatical_range": 0,
        "pronunciation": 0,
    }
    
    for sample in samples:
        for key in avg_scores:
            avg_scores[key] += sample["scores"][key]
            
    for key in avg_scores:
        avg_scores[key] /= len(samples)
        print(f"Average {key}: {avg_scores[key]:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare IELTS training dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/processed",
        help="Directory containing processed JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/ielts_training_data.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the dataset after creation",
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_dataset(args.input_dir, args.output)
    
    # Validate if requested
    if args.validate:
        validate_dataset(args.output)
