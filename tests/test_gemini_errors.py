import unittest
from unittest.mock import patch

import httpx
from google import genai

from src.providers.gemini import GeminiProvider, format_api_error, safe_retry_after


class GeminiErrorTests(unittest.TestCase):
    def call_with_error(self, status, body, headers=None, expected_attempts=1):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(status, json=body, headers=headers)

        original_client = genai.Client
        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            def create_client(**kwargs):
                options = kwargs.pop("http_options")
                self.assertEqual(options["retry_options"], {"attempts": 1, "http_status_codes": [500, 502, 503, 504]})
                return original_client(**kwargs, http_options={**options, "httpx_client": http_client})

            with patch.dict("os.environ", {"GEMINI_API_KEY": "private-test-key", "GEMINI_MODEL": "test-model"}), patch(
                "src.providers.gemini.load_dotenv"
            ), patch("src.providers.gemini.genai.Client", side_effect=create_client):
                result = GeminiProvider().generate("Private question", "Private context", "Instructions", ["45"])
        self.assertEqual(len(requests), expected_attempts)
        self.assertIsNone(result.answer)
        self.assertIsNotNone(result.total_latency_ms)
        return result

    def test_429_exposes_safe_quota_details_without_retrying(self):
        result = self.call_with_error(429, {"error": {
            "code": 429, "status": "RESOURCE_EXHAUSTED",
            "message": "Quota failure: private-test-key Bearer private-auth",
            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "QUOTA_EXCEEDED",
                         "metadata": {"authorization": "private-auth", "request": "Private context"}}],
        }}, {"Retry-After": "30", "Authorization": "Bearer private-auth"})
        self.assertEqual(result.error, "Gemini request failed (RateLimitError; status=429; reason=RESOURCE_EXHAUSTED,QUOTA_EXCEEDED; retry_after=30).")
        for secret in ["private-test-key", "private-auth", "Private question", "Private context", "authorization"]:
            self.assertNotIn(secret, result.error)

    def test_other_api_error_exposes_only_type_and_status(self):
        result = self.call_with_error(500, {"error": {"code": 500, "message": "private-test-key", "status": "INTERNAL"}}, expected_attempts=2)
        self.assertEqual(result.error, "Gemini request failed (InternalServerError; status=500).")

    def test_retry_info_is_used_without_retry_after_header(self):
        result = self.call_with_error(429, {"error": {
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12.5s"}],
        }})
        self.assertIn("retry_after=12.5s", result.error)

    def test_untrusted_diagnostic_values_are_not_echoed(self):
        result = self.call_with_error(429, {"error": {
            "status": "private-test-key", "reason": "Bearer private-auth",
            "details": [{"reason": "Private context"}],
        }}, {"Retry-After": "Bearer private-auth"})
        self.assertEqual(result.error, "Gemini request failed (RateLimitError; status=429).")

    def test_http_429_without_special_error_class(self):
        error = RuntimeError("Secret message")
        error.code = 429
        error.response_json = {"error": {"status": "RESOURCE_EXHAUSTED"}}
        error.retry_after = 10
        self.assertEqual(format_api_error(error), "Gemini request failed (RuntimeError; status=429; reason=RESOURCE_EXHAUSTED; retry_after=10).")

    def test_retry_after_http_date(self):
        self.assertEqual(safe_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), "Wed, 21 Oct 2015 07:28:00 +0000")
        self.assertIsNone(safe_retry_after({"secret": "private-test-key"}))


if __name__ == "__main__":
    unittest.main()
