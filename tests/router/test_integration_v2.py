import unittest
from app.router.router_final import RouterFinal
from app.core.models import UserRequest
from app.utils.config_loader import ConfigLoader

class TestRouterFinalIntegration(unittest.TestCase):
    def test_router_initialization_and_route(self):
        # Use existing config files
        loader = ConfigLoader(
            "config/action_catalog.yaml",
            "config/keyword_rules.yaml"
        )
        
        # Initialize RouterFinal (enable_v2=True)
        # This will use EmbedAnythingEngineFinal (Mock mode if lib missing)
        router = RouterFinal(loader, enable_v2=True)
        
        # Test routing
        req = UserRequest(text="xin nghỉ phép")
        output = router.route(req)
        
        self.assertIsNotNone(output)
        self.assertGreater(len(output.top_actions), 0)
        
        # Verify stats
        stats = router.get_stats()
        self.assertTrue(stats["enable_v2"])
        
        router.shutdown()

if __name__ == '__main__':
    unittest.main()
