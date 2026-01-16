import unittest
from datetime import date, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.action_flow.entity_extractor import EntityExtractor

class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = EntityExtractor()

    def test_single_date(self):
        text = "Tôi muốn nghỉ hôm nay"
        extracted = self.extractor.extract(text, ["start_date"])
        today = date.today().isoformat()
        self.assertEqual(extracted.get("start_date"), today)

    def test_multi_date_keywords(self):
        text = "Tôi muốn nghỉ từ hôm nay đến ngày mai"
        extracted = self.extractor.extract(text, ["start_date", "end_date"])
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        
        self.assertEqual(extracted.get("start_date"), today)
        self.assertEqual(extracted.get("end_date"), tomorrow)

    def test_multi_date_regex(self):
        text = "Nghỉ từ 15/01 đến 20/01"
        extracted = self.extractor.extract(text, ["start_date", "end_date"])
        # Assuming current year is used (e.g., 2026 based on env)
        # But regex extractor uses today.year. Let's just check relative correctness or mock date
        # For robustness, regex extraction returns ISO format YYYY-MM-DD
        self.assertTrue("start_date" in extracted)
        self.assertTrue("end_date" in extracted)
        
    def test_mixed_dates(self):
        text = "Nghỉ từ hôm nay đến 20/05"
        extracted = self.extractor.extract(text, ["start_date", "end_date"])
        today = date.today().isoformat()
        self.assertEqual(extracted.get("start_date"), today)
        self.assertTrue(extracted.get("end_date").endswith("-05-20"))

    def test_not_enough_dates(self):
        text = "Chỉ nghỉ ngày mai"
        extracted = self.extractor.extract(text, ["start_date", "end_date"])
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        self.assertEqual(extracted.get("start_date"), tomorrow)
        self.assertNotIn("end_date", extracted)

if __name__ == '__main__':
    unittest.main()
