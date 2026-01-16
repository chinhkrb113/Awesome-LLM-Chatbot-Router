import sys
import os
import asyncio
import datetime
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from app.router.router_final import RouterFinal
from app.core.models import ConversationContext
from app.utils.config_loader import ConfigLoader

class TestMemoryCleanup(unittest.TestCase):
    
    def setUp(self):
        # Mock ConfigLoader to return fast cleanup interval
        self.mock_loader = MagicMock(spec=ConfigLoader)
        self.mock_loader.get_memory_config.return_value = {
            "context_ttl_seconds": 1,        # Expire after 1s
            "cleanup_interval_seconds": 2    # Run cleanup every 2s
        }
        self.mock_loader.get_weights.return_value = {}
        
        # Mock Embedding Engine to avoid loading heavy models
        self.mock_embed_config = MagicMock()
        
        # We need to patch EmbedAnythingEngineFinal inside router_final or mock it
        # For simplicity, we assume RouterFinal handles mock injection or we just test logic directly
        
    def test_cleanup_logic_direct(self):
        """Test the cleanup logic directly without async loop first"""
        router = RouterFinal(self.mock_loader, enable_v2=False) # Skip v2 init for speed
        
        # Add contexts
        router._contexts["active"] = ConversationContext(session_id="active")
        router._contexts["expired"] = ConversationContext(session_id="expired")
        
        # Hack timestamp for expired
        router._contexts["expired"].last_updated_at = datetime.datetime.now() - datetime.timedelta(seconds=10)
        
        # Run cleanup
        router.cleanup_expired_contexts()
        
        self.assertIn("active", router._contexts)
        self.assertNotIn("expired", router._contexts)
        print("✅ Direct cleanup logic passed")

    async def async_test_background_task(self):
        """Test the background task running"""
        router = RouterFinal(self.mock_loader, enable_v2=False)
        
        # Add context
        router._contexts["expired_soon"] = ConversationContext(session_id="expired_soon")
        
        # Start task manually if not started
        if not router._cleanup_task:
            router._start_cleanup_task()
            
        print("⏳ Waiting for cleanup task (3s)...")
        await asyncio.sleep(3.5) # Wait for TTL (1s) + Interval (2s)
        
        if "expired_soon" not in router._contexts:
            print("✅ Background cleanup task passed")
        else:
            print("❌ Background cleanup task failed (Context still exists)")
            
        router.shutdown()

def run_async_test():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    test = TestMemoryCleanup()
    test.setUp()
    loop.run_until_complete(test.async_test_background_task())

if __name__ == "__main__":
    # Run sync test
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryCleanup)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Run async test
    print("\n--- Running Async Test ---")
    try:
        run_async_test()
    except Exception as e:
        print(f"Async test error: {e}")
