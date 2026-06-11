import unittest

from research.news_context_engine import (
    build_news_context,
    score_news_item,
    summarize_news_context,
)


class NewsContextEngineTests(unittest.IsolatedAsyncioTestCase):
    def test_score_symbol_negative_news_flags_risk(self):
        item = score_news_item(
            {
                "source": "Test",
                "title": "SEC lawsuit hits Solana after exchange delist risk",
                "url": "https://example.test/news",
                "symbols": ["SOL"],
            },
            "SOL",
        )

        self.assertGreaterEqual(item.relevance_score, 1.0)
        self.assertLess(item.sentiment_score, 0)
        self.assertIn("negative_symbol_news", item.risk_flags)

    def test_summarize_news_context_creates_risk_score(self):
        items = [
            score_news_item({"title": "Solana outage sparks selloff", "symbols": ["SOL"]}, "SOL"),
            score_news_item({"title": "Bitcoin market rally continues", "symbols": ["BTC"]}, "SOL"),
        ]

        context = summarize_news_context("SOL", items)

        self.assertEqual(context.symbol, "SOL")
        self.assertGreater(context.relevant_item_count, 0)
        self.assertGreater(context.risk_score, 0)
        self.assertIn("negative_symbol_news", context.risk_flags)

    async def test_build_news_context_never_crashes_on_fetch_error(self):
        async def failing_fetcher(symbol, limit):
            raise RuntimeError("network down")

        context = await build_news_context("ADA", fetcher=failing_fetcher)

        self.assertEqual(context.symbol, "ADA")
        self.assertIn("error:RuntimeError", context.provider_status)
        self.assertIn("news_context_unavailable", context.risk_flags)


if __name__ == "__main__":
    unittest.main()
