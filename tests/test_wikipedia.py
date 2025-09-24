"""
Tests for the modern Wikipedia client.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from wikipedia_async import WikipediaClient, ClientConfig
from wikipedia_async.exceptions import (
    PageNotFoundError,
    ValidationError,
    NetworkError,
    RateLimitError
)


class TestClientConfig:
    """Test ClientConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ClientConfig()
        assert config.language == "en"
        assert config.rate_limit_calls == 10
        assert config.enable_cache is True
        assert config.timeout == 30.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ClientConfig(
            language="fr",
            rate_limit_calls=20,
            timeout=60.0
        )
        assert config.language == "fr"
        assert config.rate_limit_calls == 20
        assert config.timeout == 60.0
    
    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            ClientConfig(rate_limit_calls=0)
        
        with pytest.raises(ValueError):
            ClientConfig(timeout=-1)
    
    def test_api_url_property(self):
        """Test API URL generation."""
        config = ClientConfig(language="fr")
        assert "fr.wikipedia.org" in config.api_url
    
    def test_with_language(self):
        """Test language switching."""
        config = ClientConfig(language="en")
        fr_config = config.with_language("fr")
        
        assert config.language == "en"
        assert fr_config.language == "fr"
        assert config is not fr_config


class TestWikipediaClient:
    """Test WikipediaClient class."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        config = ClientConfig(
            rate_limit_calls=100,  # High limit for tests
            cache_ttl=1,           # Short cache for tests
        )
        return WikipediaClient(config=config)
    
    @pytest.fixture
    def mock_response(self):
        """Mock API response."""
        return {
            "query": {
                "pages": [{
                    "pageid": 12345,
                    "title": "Test Page",
                    "fullurl": "https://en.wikipedia.org/wiki/Test_Page",
                    "extract": "This is a test page."
                }]
            }
        }
    
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.config is not None
        assert client._rate_limiter is not None
        assert client._cache is not None
    
    async def test_context_manager(self):
        """Test async context manager."""
        async with WikipediaClient() as client:
            assert client._session is not None
        # Session should be closed after context
    
    @patch('aiohttp.ClientSession.get')
    async def test_search(self, mock_get, client):
        """Test search functionality."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "query": {
                "search": [{
                    "title": "Test Result",
                    "snippet": "Test snippet",
                    "pageid": 123
                }]
            }
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        results = await client.search("test query")
        
        assert len(results) > 0
        assert results[0].title == "Test Result"
        assert results[0].snippet == "Test snippet"
    
    async def test_search_validation(self, client):
        """Test search input validation."""
        with pytest.raises(ValidationError):
            await client.search("")  # Empty query
        
        with pytest.raises(ValidationError):
            await client.search("test", limit=0)  # Invalid limit
    
    @patch('aiohttp.ClientSession.get')
    async def test_get_page(self, mock_get, client, mock_response):
        """Test get_page functionality."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_get.return_value.__aenter__.return_value = mock_resp
        
        page = await client.get_page("Test Page")
        
        assert page.title == "Test Page"
        assert page.page_id == 12345
        assert "test page" in page.extract.lower()
    
    async def test_get_page_validation(self, client):
        """Test get_page input validation."""
        with pytest.raises(ValidationError):
            await client.get_page()  # No title or page_id
        
        with pytest.raises(ValidationError):
            await client.get_page(title="test", page_id=123)  # Both provided
    
    @patch('aiohttp.ClientSession.get')
    async def test_page_not_found(self, mock_get, client):
        """Test page not found error."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "query": {
                "pages": [{
                    "missing": True,
                    "title": "NonexistentPage"
                }]
            }
        })
        mock_get.return_value.__aenter__.return_value = mock_resp
        
        with pytest.raises(PageNotFoundError):
            await client.get_page("NonexistentPage")
    
    @patch('aiohttp.ClientSession.get')
    async def test_network_error(self, mock_get, client):
        """Test network error handling."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.reason = "Internal Server Error"
        mock_get.return_value.__aenter__.return_value = mock_resp
        
        with pytest.raises(NetworkError):
            await client.search("test")
    
    @patch('aiohttp.ClientSession.get')
    async def test_rate_limit_error(self, mock_get, client):
        """Test rate limit error handling."""
        mock_resp = AsyncMock()
        mock_resp.status = 429
        mock_resp.headers = {"Retry-After": "5"}
        mock_get.return_value.__aenter__.return_value = mock_resp
        
        with pytest.raises(RateLimitError) as exc_info:
            await client.search("test")
        
        assert exc_info.value.retry_after == 5.0
    
    async def test_geosearch_validation(self, client):
        """Test geosearch input validation."""
        with pytest.raises(ValidationError):
            await client.geosearch(latitude=91, longitude=0)  # Invalid latitude
        
        with pytest.raises(ValidationError):
            await client.geosearch(latitude=0, longitude=181)  # Invalid longitude
        
        with pytest.raises(ValidationError):
            await client.geosearch(latitude=0, longitude=0, radius=5)  # Invalid radius
    
    @patch('aiohttp.ClientSession.get')
    async def test_batch_operations(self, mock_get, client):
        """Test batch operations."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "query": {
                "pages": [
                    {
                        "pageid": 1,
                        "title": "Page 1",
                        "fullurl": "https://en.wikipedia.org/wiki/Page_1",
                        "extract": "First page"
                    },
                    {
                        "pageid": 2,
                        "title": "Page 2", 
                        "fullurl": "https://en.wikipedia.org/wiki/Page_2",
                        "extract": "Second page"
                    }
                ]
            }
        })
        mock_get.return_value.__aenter__.return_value = mock_resp
        
        titles = ["Page 1", "Page 2"]
        result = await client.get_pages_batch(titles)
        
        assert result.total_requested == 2
        assert len(result.successful) == 2
        assert len(result.failed) == 0
        assert result.success_rate == 100.0
    
    async def test_language_switching(self, client):
        """Test language switching."""
        original_lang = client.config.language
        
        await client.set_language("fr")
        
        assert client.config.language == "fr"
        assert client.config.language != original_lang
    
    async def test_caching(self, client):
        """Test caching functionality."""
        # Test cache stats
        stats = await client.cache_stats()
        assert stats["cache_enabled"] is True
        
        # Test cache clear
        await client.clear_cache()
        stats_after_clear = await client.cache_stats()
        assert stats_after_clear["total_entries"] == 0


class TestModels:
    """Test Pydantic models."""
    
    def test_coordinates_validation(self):
        """Test Coordinates model validation."""
        from wikipedia_async.models import Coordinates
        from decimal import Decimal
        
        # Valid coordinates
        coords = Coordinates(latitude=Decimal("48.8566"), longitude=Decimal("2.3522"))
        assert coords.latitude == Decimal("48.8566")
        
        # Invalid latitude
        with pytest.raises(ValueError):
            Coordinates(latitude=Decimal("91"), longitude=Decimal("0"))
        
        # Invalid longitude
        with pytest.raises(ValueError):
            Coordinates(latitude=Decimal("0"), longitude=Decimal("181"))
    
    def test_search_result_model(self):
        """Test SearchResult model."""
        from wikipedia_async.models import SearchResult
        
        result = SearchResult(
            title="Test Page",
            snippet="Test snippet",
            page_id=123
        )
        
        assert result.title == "Test Page"
        assert result.page_id == 123
    
    def test_wiki_page_model(self):
        """Test WikiPage model."""
        from wikipedia_async.models import WikiPage
        
        page = WikiPage(
            title="Test Page",
            page_id=123,
            url="https://en.wikipedia.org/wiki/Test_Page"
        )
        
        assert page.title == "Test Page"
        assert page.page_id == 123
        assert page.language == "en"  # Default value
    
    def test_batch_result_success_rate(self):
        """Test BatchResult success rate calculation."""
        from wikipedia_async.models import BatchResult, WikiPage
        
        # All successful
        result = BatchResult(
            successful=[
                WikiPage(title="Page 1", page_id=1, url="http://example.com/1"),
                WikiPage(title="Page 2", page_id=2, url="http://example.com/2")
            ],
            failed=[],
            total_requested=2
        )
        assert result.success_rate == 100.0
        
        # Partial success
        result = BatchResult(
            successful=[
                WikiPage(title="Page 1", page_id=1, url="http://example.com/1")
            ],
            failed=[{"title": "Page 2", "error": "Not found"}],
            total_requested=2
        )
        assert result.success_rate == 50.0
        
        # No requests
        result = BatchResult(
            successful=[],
            failed=[],
            total_requested=0
        )
        assert result.success_rate == 0.0


# Performance tests
class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request handling."""
        config = ClientConfig(max_concurrent_requests=5)
        
        async with WikipediaClient(config=config) as client:
            # Mock the _make_request method to simulate API calls
            original_make_request = client._make_request
            
            async def mock_make_request(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate API delay
                return {"query": {"search": []}}
            
            client._make_request = mock_make_request
            
            # Start multiple concurrent searches
            tasks = [
                client.search(f"query {i}")
                for i in range(10)
            ]
            
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            # Should complete faster than sequential execution
            assert len(results) == 10
            assert end_time - start_time < 1.0  # Should be much less than 1 second


# Integration tests (require network)
class TestIntegration:
    """Integration tests with real API calls."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_search(self):
        """Test with real Wikipedia API."""
        async with WikipediaClient() as client:
            results = await client.search("Python programming", limit=5)
            
            assert len(results) > 0
            assert any("python" in result.title.lower() for result in results)
    
    @pytest.mark.integration 
    @pytest.mark.asyncio
    async def test_real_page_fetch(self):
        """Test fetching real page."""
        async with WikipediaClient() as client:
            page = await client.get_page("Python (programming language)")
            
            assert page.title
            assert page.page_id
            assert page.extract
            assert "python" in page.extract.lower()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_geosearch(self):
        """Test real geographic search."""
        async with WikipediaClient() as client:
            # Search near London
            results = await client.geosearch(
                latitude=51.5074,
                longitude=-0.1278,
                radius=1000,
                limit=5
            )
            
            assert len(results) > 0
            assert all(result.coordinates for result in results)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])