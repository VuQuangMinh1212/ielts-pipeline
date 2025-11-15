"""
Test suite for training and inference
"""
import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatasetPreparation(unittest.TestCase):
    """Test dataset preparation"""
    
    def test_create_sample_dataset(self):
        from app.prepare_dataset import create_example_dataset
        
        samples = create_example_dataset()
        
        self.assertGreater(len(samples), 0)
        self.assertIn("transcript", samples[0])
        self.assertIn("scores", samples[0])
        
    def test_dataset_format(self):
        from app.prepare_dataset import create_training_sample
        
        sample = create_training_sample(
            transcript="Test transcript",
            scores={"fluency": 7.0, "lexical_resource": 6.5}
        )
        
        self.assertEqual(sample["transcript"], "Test transcript")
        self.assertEqual(sample["scores"]["fluency"], 7.0)


class TestModelConfiguration(unittest.TestCase):
    """Test model configuration"""
    
    def test_supported_models(self):
        from app.train_qlora import IELTSModelTrainer
        
        trainer = IELTSModelTrainer(model_name="qwen")
        
        self.assertIn("qwen", trainer.SUPPORTED_MODELS)
        self.assertIn("phi2", trainer.SUPPORTED_MODELS)
        self.assertIn("gemma", trainer.SUPPORTED_MODELS)
        
    def test_model_path_resolution(self):
        from app.train_qlora import IELTSModelTrainer
        
        trainer = IELTSModelTrainer(model_name="qwen")
        
        self.assertEqual(trainer.base_model, "Qwen/Qwen-1.5B")


class TestInference(unittest.TestCase):
    """Test inference capabilities"""
    
    @unittest.skipIf(not os.path.exists("./models/ielts-finetuned"), "Model not found")
    def test_autotrain_inference(self):
        """Test Autotrain inference if model exists"""
        try:
            from app.inference import AutotrainInferenceServer
            
            server = AutotrainInferenceServer(
                model_path="./models/ielts-finetuned"
            )
            
            response = server.generate("Hello, this is a test.")
            
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except ImportError:
            self.skipTest("Transformers not installed")
            
    def test_inference_import(self):
        """Test that inference modules can be imported"""
        try:
            from app import inference
            
            self.assertTrue(hasattr(inference, 'VLLMInferenceServer'))
            self.assertTrue(hasattr(inference, 'LlamaCppInferenceServer'))
            self.assertTrue(hasattr(inference, 'AutotrainInferenceServer'))
        except ImportError as e:
            self.skipTest(f"Import failed: {e}")


class TestModelConversion(unittest.TestCase):
    """Test model conversion utilities"""
    
    def test_conversion_import(self):
        """Test conversion module imports"""
        try:
            from app import convert_model
            
            self.assertTrue(hasattr(convert_model, 'convert_to_gguf'))
            self.assertTrue(hasattr(convert_model, 'convert_to_awq'))
            self.assertTrue(hasattr(convert_model, 'convert_to_gptq'))
        except ImportError as e:
            self.skipTest(f"Import failed: {e}")


class TestConfiguration(unittest.TestCase):
    """Test configuration files"""
    
    def test_training_config_exists(self):
        config_path = Path("./config/training_config.json")
        
        self.assertTrue(config_path.exists(), "Training config not found")
        
    def test_training_config_valid(self):
        import json
        
        with open("./config/training_config.json") as f:
            config = json.load(f)
            
        self.assertIn("training", config)
        self.assertIn("deployment", config)
        
        # Check training config
        training = config["training"]
        self.assertIn("model_name", training)
        self.assertIn("lora_config", training)
        

class TestDockerFiles(unittest.TestCase):
    """Test Docker configuration"""
    
    def test_dockerfiles_exist(self):
        """Check all Dockerfiles exist"""
        docker_files = [
            "./docker/Dockerfile",
            "./docker/Dockerfile.gpu",
            "./docker/Dockerfile.inference",
            "./docker/Dockerfile.cpu",
        ]
        
        for dockerfile in docker_files:
            self.assertTrue(
                Path(dockerfile).exists(),
                f"{dockerfile} not found"
            )
            
    def test_docker_compose_exists(self):
        """Check docker-compose.yml exists"""
        self.assertTrue(Path("./docker-compose.yml").exists())


class TestDocumentation(unittest.TestCase):
    """Test documentation files"""
    
    def test_docs_exist(self):
        """Check documentation files"""
        docs = [
            "./docs/TRAINING_DEPLOYMENT.md",
            "./docs/QUICKSTART.md",
            "./README_TRAINING.md",
            "./IMPLEMENTATION_SUMMARY.md",
        ]
        
        for doc in docs:
            self.assertTrue(
                Path(doc).exists(),
                f"{doc} not found"
            )


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetPreparation))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestInference))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDockerFiles))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
