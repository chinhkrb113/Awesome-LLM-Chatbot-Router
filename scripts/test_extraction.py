import sys
import os
import datetime
# Add project root to path
sys.path.append(os.getcwd())

from app.action_flow.entity_extractor import EntityExtractor

def test_extraction():
    extractor = EntityExtractor()
    today = datetime.date.today()
    
    test_cases = [
        {
            "text": "mai tôi xin nghỉ về quê có việc gia đình",
            "slots": ["leave_type", "start_date"],
            "expected": {
                "leave_type": "Nghỉ việc riêng",
                # start_date check logic below
            }
        },
        {
            "text": "tôi bị sốt, xin nghỉ ốm hôm nay",
            "slots": ["leave_type", "start_date"],
            "expected": {
                "leave_type": "Nghỉ ốm",
                "start_date": today.isoformat()
            }
        },
        {
            "text": "xin nghỉ phép năm đi du lịch 3 ngày",
            "slots": ["leave_type", "duration"],
            "expected": {
                "leave_type": "Nghỉ phép năm",
                "duration": "3"
            }
        },
        {
            "text": "cho anh Nguyễn Văn A vào cổng ngày mai",
            "slots": ["visitor_name", "visit_date"],
            "expected": {
                "visitor_name": "Nguyễn Văn A"
            }
        }
    ]
    
    print("=== STARTING EXTRACTION TESTS ===")
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: '{case['text']}'")
        result = extractor.extract(case['text'], case['slots'])
        
        # Check expected values
        for k, v in case['expected'].items():
            actual = result.get(k)
            if actual == v:
                print(f"  ✅ {k}: {actual}")
            else:
                print(f"  ❌ {k}: Expected '{v}', Got '{actual}'")
                
        # Special check for 'tomorrow' logic
        if "mai" in case['text'] and 'start_date' in result:
             tomorrow = (today + datetime.timedelta(days=1)).isoformat()
             if result['start_date'] == tomorrow:
                 print(f"  ✅ start_date (tomorrow): {tomorrow}")
             else:
                 print(f"  ❌ start_date: Expected {tomorrow}, Got {result['start_date']}")

    print("\n=== TESTS COMPLETE ===")

if __name__ == "__main__":
    test_extraction()
