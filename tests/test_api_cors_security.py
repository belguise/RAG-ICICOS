"""
Automated Security Tests for CORS and API Access Control.
Verifies allowed development/production origins, origin regex matching,
credential handling, and rejection of unauthorized origins.
"""

import unittest
from fastapi.testclient import TestClient
from agentic_rag_sop.api.main import app


class TestAPICORSSecurity(unittest.TestCase):
    """Test suite verifying CORS security configuration on FastAPI app."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_allowed_origin_localhost_3000(self):
        """Allowed origin (Next.js local dev) must receive matching Access-Control headers."""
        headers = {
            "Origin": "http://localhost:3000",
        }
        response = self.client.get("/health", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:3000"
        )
        self.assertEqual(
            response.headers.get("access-control-allow-credentials"),
            "true"
        )

    def test_allowed_origin_localhost_5173(self):
        """Allowed origin (Vite local dev) must receive matching Access-Control headers."""
        headers = {
            "Origin": "http://localhost:5173",
        }
        response = self.client.get("/health", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173"
        )

    def test_allowed_origin_vercel_project_regex(self):
        """Specific project Vercel deployments matching regex must be allowed."""
        headers = {
            "Origin": "https://rag-icicos.vercel.app",
        }
        response = self.client.get("/health", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "https://rag-icicos.vercel.app"
        )

    def test_rejected_unauthorized_origin(self):
        """Unauthorized external origin must NOT receive access-control-allow-origin header."""
        headers = {
            "Origin": "https://malicious-attacker.com",
        }
        response = self.client.get("/health", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_rejected_arbitrary_vercel_origin(self):
        """Arbitrary non-project vercel app must NOT match restricted origin regex."""
        headers = {
            "Origin": "https://unrelated-phishing-app.vercel.app",
        }
        response = self.client.get("/health", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_preflight_options_request_allowed_origin(self):
        """Preflight OPTIONS request from allowed origin returns proper headers."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, Authorization",
        }
        response = self.client.options("/api/v1/chat", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:3000"
        )
        self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))

    def test_preflight_options_request_rejected_origin(self):
        """Preflight OPTIONS request from unauthorized origin does NOT receive allow-origin."""
        headers = {
            "Origin": "https://evil.org",
            "Access-Control-Request-Method": "POST",
        }
        response = self.client.options("/api/v1/chat", headers=headers)
        self.assertNotIn("access-control-allow-origin", response.headers)


if __name__ == "__main__":
    unittest.main()
