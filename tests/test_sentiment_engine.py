import unittest
from unittest.mock import Mock, patch

from tools.sentiment_engine import fear_greed_to_feature, get_fear_greed_index


class SentimentEngineTests(unittest.TestCase):
    def test_fear_greed_fear_is_positive_contrarian_feature(self):
        self.assertGreater(fear_greed_to_feature(10), 0)

    def test_fear_greed_greed_is_negative_contrarian_feature(self):
        self.assertLess(fear_greed_to_feature(90), 0)

    def test_fear_greed_neutral_is_zero(self):
        self.assertEqual(fear_greed_to_feature(50), 0.0)

    def test_get_fear_greed_index_returns_expected_dict(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [{
                "value": "23",
                "value_classification": "Extreme Fear",
                "timestamp": "1716163200",
            }]
        }
        with patch("tools.sentiment_engine.httpx.get", return_value=response):
            result = get_fear_greed_index()

        self.assertEqual(result, {
            "value": 23,
            "classification": "Extreme Fear",
            "timestamp": "1716163200",
        })


if __name__ == "__main__":
    unittest.main()
