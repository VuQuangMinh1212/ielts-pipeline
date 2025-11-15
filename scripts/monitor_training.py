"""
Monitor training progress
"""
import time
import subprocess
import sys

def monitor_training():
    """Monitor Docker container training logs"""
    print("🔍 Monitoring IELTS Training...")
    print("=" * 60)
    
    try:
        # Follow logs from training container
        process = subprocess.Popen(
            ["docker", "logs", "-f", "ielts-trainer"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            
            # Check for completion
            if "Training complete" in line or "Saving model" in line:
                print("\n✅ Training finished!")
                break
                
            # Check for errors
            if "ERROR" in line or "FAILED" in line:
                print("\n❌ Training error detected!")
                
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error monitoring: {e}")
        

def get_container_status():
    """Get training container status"""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=ielts-trainer", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except:
        return "Unknown"


if __name__ == "__main__":
    status = get_container_status()
    print(f"Container Status: {status}\n")
    
    if "Up" in status:
        monitor_training()
    else:
        print("⚠️ Training container is not running")
        print("Start training with: docker-compose --profile training up trainer")
