import unittest
from models import ZeroShotModels

class TestZeroShotModels(unittest.TestCase):

    def setUp(self):
        # Create an instance of the ZeroShotModels class before running each test
        self.zero_shot_models = ZeroShotModels()

    def test_get_models_by_suffix(self):
        # Test if the method returns the correct list of models with the given suffix
        expected_models = [
            {
                "model": None,
                "suffix": "DeBERTa",
                "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                "context_size": 800,
            },
            # Add other expected models here based on your test data
        ]
        suffix = "DeBERTa"
        models = self.zero_shot_models.get_models_by_suffix(suffix)
        self.assertEqual(models, expected_models)

    def test_get_models_by_name(self):
        # Test if the method returns the correct list of models with the given model name
        expected_models = [
            {
                "model": None,
                "suffix": "DeBERTa",
                "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                "context_size": 800,
            },
            # Add other expected models here based on your test data
        ]
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        models = self.zero_shot_models.get_models_by_name(model_name)
        self.assertEqual(models, expected_models)

    def test_get_all_suffixes(self):
        # Test if the method returns all unique suffixes in the models
        expected_suffixes = ["DeBERTa", "facebook_zero", "mpt-7b", "llama-2-7b", "llama-2-13b", "falcon-7b", "vicuna"]
        suffixes = self.zero_shot_models.get_all_suffixes()
        self.assertEqual(sorted(suffixes), sorted(expected_suffixes))

    # Add more test cases for other methods if needed

if __name__ == "__main__":
    unittest.main()
