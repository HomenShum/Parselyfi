"""
Ethical Web Scraper with Redirect Handling

This module enhances the previous scraper with redirect handling capabilities
to properly follow redirection URLs and scrape their final destinations.
"""

import asyncio
import aiohttp
import logging
import time
import hashlib
import re
import json
import socket
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import trafilatura, but provide a fallback if not available
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    logger.warning("Trafilatura not installed. Will use BeautifulSoup fallback for content extraction.")
    TRAFILATURA_AVAILABLE = False

@dataclass
class ScraperConfig:
    """Configuration for the web scraper"""
    request_delay: float = 3.0  # Base delay between requests (seconds)
    timeout: int = 30  # Request timeout (seconds)
    max_retries: int = 3  # Maximum retry attempts
    user_agent: str = "Mozilla/5.0 (compatible; EthicalScraper/1.0; +https://github.com/ethical-content-analysis)"
    max_concurrent: int = 5  # Maximum concurrent requests
    respect_robots: bool = True  # Whether to respect robots.txt
    cache_enabled: bool = True  # Enable caching of responses
    content_size_limit: int = 10 * 1024 * 1024  # 10MB content size limit
    follow_redirects: bool = True  # Whether to follow redirects
    max_redirect_depth: int = 5  # Maximum redirect depth to follow
    javascript_rendering: bool = False  # Enable JavaScript rendering (requires playwright)
    proxy: Optional[str] = None  # Proxy server URL

class EthicalWebScraper:
    """
    Ethical web scraper with enhanced redirect handling
    
    Features:
    - Follows and resolves redirect chains
    - Respects robots.txt directives
    - Implements rate limiting and exponential backoff
    - DNS and response caching
    - Content extraction with fallbacks
    - Error handling and retry logic
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize the scraper with configuration
        
        Args:
            config: ScraperConfig object with scraper settings
        """
        self.config = config if config else ScraperConfig()
        self.domain_delays: Dict[str, float] = {}
        self.last_request_times: Dict[str, float] = {}
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Tuple[float, str]] = {}  # URL: (timestamp, content)
        self.redirect_cache: Dict[str, Tuple[float, str]] = {}  # Original URL: (timestamp, final URL)
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.blacklist = self._load_blacklist()
        
        # Known redirect services
        self.redirect_services = [
            "vertexaisearch.cloud.google.com",
            "bit.ly",
            "tinyurl.com",
            "t.co",
            "goo.gl",
            "ow.ly",
            "is.gd",
            "buff.ly",
        ]
        
        # Initialize DNS cache
        self.dns_cache: Dict[str, str] = {}
        socket.setdefaulttimeout(self.config.timeout)
        
    def _load_blacklist(self) -> List[str]:
        """Load domain blacklist from file if exists"""
        try:
            with open('blacklist.txt', 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return []
    
    async def __aenter__(self):
        """Async context manager entry"""
        headers = {
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/',
            'DNT': '1'
        }
        
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent,
            force_close=True,
            enable_cleanup_closed=True,
            ttl_dns_cache=300
        )
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            trust_env=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc
    
    def _is_blacklisted(self, url: str) -> bool:
        """Check if URL is in blacklist"""
        domain = self._get_domain(url)
        return any(blacklisted in domain for blacklisted in self.blacklist)
    
    def _is_redirect_service(self, url: str) -> bool:
        """Check if URL is from a known redirect service"""
        domain = self._get_domain(url)
        return any(service in domain for service in self.redirect_services)
    
    async def _resolve_redirects(self, url: str, depth: int = 0) -> str:
        """
        Resolve redirect chains to get the final URL
        
        Args:
            url: The starting URL
            depth: Current redirect depth
            
        Returns:
            The final URL after following redirects
        """
        if depth >= self.config.max_redirect_depth:
            logger.warning(f"Max redirect depth ({self.config.max_redirect_depth}) reached for {url}")
            return url
            
        # Check redirect cache
        if self.config.cache_enabled and url in self.redirect_cache:
            timestamp, final_url = self.redirect_cache[url]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Using cached redirect for {url} -> {final_url}")
                return final_url
        
        # If not a redirect service, no need to resolve
        if not self._is_redirect_service(url):
            return url
            
        try:
            # Use HEAD request to avoid downloading content
            async with self.session.head(
                url, 
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=10)  # Use shorter timeout for redirect checks
            ) as response:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if location:
                        # Handle relative redirects
                        if not location.startswith(('http://', 'https://')):
                            location = urljoin(url, location)
                        
                        logger.debug(f"Following redirect: {url} -> {location}")
                        
                        # Recursively follow redirects
                        final_url = await self._resolve_redirects(location, depth + 1)
                        
                        # Cache the result
                        if self.config.cache_enabled:
                            self.redirect_cache[url] = (time.time(), final_url)
                            
                        return final_url
                else:
                    # No redirect
                    return url
        except Exception as e:
            logger.warning(f"Error resolving redirect for {url}: {str(e)}")
            return url
    
    async def _get_robots_parser(self, url: str) -> Optional[RobotFileParser]:
        """Get or create robots.txt parser for domain"""
        if not self.config.respect_robots:
            return None
            
        domain = self._get_domain(url)
        if domain in self.robots_parsers:
            return self.robots_parsers[domain]
        
        robots_url = f"{urlparse(url).scheme}://{domain}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)  # Set the URL for the parser
        
        try:
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    content = await response.text()
                    rp.parse(content.splitlines())
                    
                    # Check for crawl delay
                    crawl_delay = rp.crawl_delay(self.config.user_agent)
                    if crawl_delay:
                        self.domain_delays[domain] = max(crawl_delay, self.config.request_delay)
                        logger.info(f"Set crawl delay for {domain}: {crawl_delay}s")
                    
                    self.robots_parsers[domain] = rp
                    return rp
                elif response.status == 404:
                    # No robots.txt file exists, which means all URLs are allowed
                    logger.info(f"No robots.txt found for {domain} (HTTP 404), all URLs are allowed")
                    rp.allow_all = True  # Set the parser to allow all URLs
                    self.robots_parsers[domain] = rp
                    return rp
                else:
                    # Other error status, default to being permissive
                    logger.warning(f"Unexpected status code {response.status} for robots.txt at {robots_url}, defaulting to allow all")
                    rp.allow_all = True
                    self.robots_parsers[domain] = rp
                    return rp
        except Exception as e:
            logger.warning(f"Could not fetch robots.txt for {domain}: {str(e)}, defaulting to allow all")
            rp.allow_all = True  # Set to allow all URLs if robots.txt cannot be fetched
            self.robots_parsers[domain] = rp
            return rp
    
    async def _enforce_delay(self, domain: str):
        """Enforce delay between requests to the same domain"""
        if domain not in self.last_request_times:
            self.last_request_times[domain] = time.time()
            return
            
        last_time = self.last_request_times[domain]
        elapsed = time.time() - last_time
        required_delay = self.domain_delays.get(domain, self.config.request_delay)
        
        if elapsed < required_delay:
            sleep_time = required_delay - elapsed
            logger.debug(f"Respecting crawl delay for {domain}: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            
        self.last_request_times[domain] = time.time()
    
    async def _resolve_dns(self, domain: str) -> str:
        """Resolve DNS with caching"""
        if domain in self.dns_cache:
            return self.dns_cache[domain]
        
        try:
            # Add some jitter to DNS requests
            await asyncio.sleep(random.uniform(0.1, 0.5))
            addr_info = await asyncio.get_event_loop().getaddrinfo(
                domain, 80, proto=socket.IPPROTO_TCP
            )
            ip = addr_info[0][4][0]
            self.dns_cache[domain] = ip
            return ip
        except Exception as e:
            logger.warning(f"DNS resolution failed for {domain}: {str(e)}")
            return domain
    
    async def _fetch_url(self, url: str, retry_count: int = 0) -> Optional[str]:
        """Fetch URL content with retries and error handling"""
        if self._is_blacklisted(url):
            logger.warning(f"URL {url} is in blacklist, skipping")
            return None
            
        # First resolve any redirects to get the final URL
        original_url = url
        if self._is_redirect_service(url):
            url = await self._resolve_redirects(url)
            logger.info(f"Resolved redirect: {original_url} -> {url}")
        
        domain = self._get_domain(url)
        
        try:
            # Check cache first
            cache_key = original_url if self.config.cache_enabled else url
            if self.config.cache_enabled and cache_key in self.cache:
                timestamp, content = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"Using cached content for {url}")
                    return content
                else:
                    del self.cache[cache_key]
            
            # Respect robots.txt and crawl delays
            if self.config.respect_robots:
                robots_parser = await self._get_robots_parser(url)
                if robots_parser and not robots_parser.can_fetch(self.config.user_agent, url):
                    logger.warning(f"URL {url} disallowed by robots.txt")
                    return None
            
            await self._enforce_delay(domain)
            
            # Resolve DNS (with caching)
            ip = await self._resolve_dns(domain)
            headers = {
                'Host': domain,
                'X-Forwarded-For': ip,
                'X-Real-IP': ip
            }
            
            # Make the request
            logger.info(f"Fetching {url} (attempt {retry_count + 1})")
            
            if self.config.proxy:
                proxy_url = self.config.proxy
            else:
                proxy_url = None
                
            async with self.session.get(
                url,
                headers=headers,
                proxy=proxy_url,
                allow_redirects=self.config.follow_redirects
            ) as response:
                # Check response status
                if response.status >= 400:
                    logger.warning(f"HTTP {response.status} for {url}")
                    if response.status == 429:  # Too Many Requests
                        retry_after = response.headers.get('Retry-After', self.config.request_delay * 2)
                        await asyncio.sleep(float(retry_after))
                        return await self._fetch_url(url, retry_count + 1)
                    return None
                
                # Check content type and size
                content_type = response.headers.get('Content-Type', '')
                content_length = int(response.headers.get('Content-Length', 0))
                
                if 'html' not in content_type.lower() and 'text' not in content_type.lower():
                    logger.warning(f"Non-HTML/text content type: {content_type}")
                    return None
                
                if content_length > self.config.content_size_limit:
                    logger.warning(f"Content too large: {content_length} bytes")
                    return None
                
                # Read content with size limit
                content = await response.text(errors='replace')
                
                if len(content) > self.config.content_size_limit:
                    logger.warning(f"Content exceeded size limit after download")
                    return None
                
                # Cache the response
                if self.config.cache_enabled:
                    self.cache[cache_key] = (time.time(), content)
                
                return content
                
        except aiohttp.ClientError as e:
            logger.warning(f"Request failed for {url}: {str(e)}")
            if retry_count < self.config.max_retries:
                retry_delay = min(2 ** retry_count, 10)  # Exponential backoff, max 10s
                await asyncio.sleep(retry_delay)
                return await self._fetch_url(url, retry_count + 1)
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {str(e)}")
            return None
    
    async def _extract_content(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Extract structured content from HTML"""
        try:
            # First try with trafilatura (fast and good for articles)
            if TRAFILATURA_AVAILABLE:
                extracted = await asyncio.to_thread(
                    trafilatura.extract,
                    html,
                    url=url,
                    include_comments=False,
                    include_tables=True,
                    include_links=True,
                    output_format='json'
                )
                
                if extracted:
                    result = json.loads(extracted)
                    
                    # Fallback to BeautifulSoup if needed
                    if not result.get('text') or len(result['text']) < 100:
                        logger.debug("Trafilatura extraction was minimal, falling back to BeautifulSoup")
                        result = await self._extract_with_beautifulsoup(html)
                    
                    # Add metadata
                    result['url'] = url
                    result['timestamp'] = time.time()
                    result['content_hash'] = hashlib.sha256(result.get('text', '').encode()).hexdigest()
                    
                    return result
            else:
                # If trafilatura is not available, use BeautifulSoup
                result = await self._extract_with_beautifulsoup(html)
                
                # Add metadata
                result['url'] = url
                result['timestamp'] = time.time()
                result['content_hash'] = hashlib.sha256(result.get('text', '').encode()).hexdigest()
                
                return result
                
            return None
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return None
            
    async def _extract_with_beautifulsoup(self, html: str) -> Dict[str, Any]:
        """Extract content using BeautifulSoup as a fallback"""
        try:
            result = {
                'text': '',
                'title': '',
                'source': 'BeautifulSoup fallback'
            }
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            if soup.title:
                result['title'] = soup.title.string
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'img', 'svg']):
                element.decompose()
            
            # Try to find main content
            article = soup.find('article') or soup.find('main') or soup.find('div', role='main')
            if article:
                text = article.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            if text and len(text) > 100:
                result['text'] = ' '.join(text.split())
            
            # Extract more metadata
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                result['description'] = meta_desc.get('content', '')
                
            return result
        except Exception as e:
            logger.error(f"Error in BeautifulSoup extraction: {str(e)}")
            return {'text': '', 'title': '', 'source': 'BeautifulSoup error'}
    
    async def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Main method to scrape a single page"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        html = await self._fetch_url(url)
        if not html:
            return None
            
        return await self._extract_content(html, url)
    
    async def scrape_with_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape page including detailed metadata"""
        html = await self._fetch_url(url)
        if not html:
            return None
            
        content = await self._extract_content(html, url)
        if not content:
            return None
            
        metadata = await self.extract_metadata(html)
        content['metadata'] = metadata
        
        return content
    
    async def extract_metadata(self, html: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {
            'title': '',
            'description': '',
            'keywords': [],
            'authors': [],
            'published_date': '',
            'canonical_url': '',
            'og_properties': {},
            'twitter_properties': {}
        }
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Basic meta tags
            if soup.title:
                metadata['title'] = soup.title.string
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata['description'] = meta_desc.get('content', '')
            
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                metadata['keywords'] = [kw.strip() for kw in meta_keywords.get('content', '').split(',')]
            
            # Authors
            meta_author = soup.find('meta', attrs={'name': 'author'})
            if meta_author:
                metadata['authors'].append(meta_author.get('content', ''))
            
            # Published date
            for prop in ['article:published_time', 'datePublished']:
                meta_date = soup.find('meta', attrs={'property': prop}) or \
                           soup.find('meta', attrs={'name': prop})
                if meta_date:
                    metadata['published_date'] = meta_date.get('content', '')
                    break
            
            # Canonical URL
            canonical = soup.find('link', rel='canonical')
            if canonical:
                metadata['canonical_url'] = canonical.get('href', '')
            
            # Open Graph properties
            og_props = {}
            for tag in soup.find_all('meta', property=re.compile(r'^og:')):
                prop_name = tag['property'][3:]  # Remove 'og:' prefix
                og_props[prop_name] = tag.get('content', '')
            metadata['og_properties'] = og_props
            
            # Twitter Card properties
            twitter_props = {}
            for tag in soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')}):
                prop_name = tag['name'][8:]  # Remove 'twitter:' prefix
                twitter_props[prop_name] = tag.get('content', '')
            metadata['twitter_properties'] = twitter_props
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
        
        return metadata

    async def check_robots_allowed(self, url: str) -> bool:
        """Check if a URL is allowed by robots.txt"""
        if not self.config.respect_robots:
            return True
            
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # First resolve any redirects to get the final URL
        original_url = url
        if self._is_redirect_service(url):
            url = await self._resolve_redirects(url)
            logger.info(f"Resolved redirect for robots check: {original_url} -> {url}")
            
        domain = self._get_domain(url)
        
        try:
            # Create a temporary session if needed
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    self.session = session
                    robots_parser = await self._get_robots_parser(url)
            else:
                robots_parser = await self._get_robots_parser(url)
                
            if not robots_parser:
                logger.info(f"No robots parser found for {url}, defaulting to allow")
                return True
                
            if robots_parser.can_fetch(self.config.user_agent, url):
                return True
            else:
                logger.warning(f"URL {url} disallowed by robots.txt")
                return False
        except Exception as e:
            logger.error(f"Error checking robots.txt: {str(e)}")
            # Default to True (allowed) on error to be less restrictive
            return True

async def scrape_urls_from_context(additional_context: str, max_concurrent: int = 3) -> str:
    """
    Extract URLs from additional context, scrape their content, and add it to the context
    
    Args:
        additional_context: Original additional context string containing URLs
        max_concurrent: Maximum number of concurrent scraping tasks
        
    Returns:
        Enhanced additional context with content from scraped URLs
    """
    import re
    import asyncio
    from urllib.parse import urlparse
    
    # Initialize the scraper
    scraper_config = ScraperConfig(
        request_delay=2.0,
        user_agent="ContentAnalysisTool/1.0 (+https://github.com/ethical-content-analysis)",
        respect_robots=True,
        cache_enabled=True,
        max_concurrent=max_concurrent,
        max_redirect_depth=5
    )
    scraper = EthicalWebScraper(scraper_config)
    
    # Extract URLs from the context
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, additional_context)
    
    if not urls:
        return additional_context
    
    # Track which URLs have been scraped to avoid duplicates
    scraped_urls = {}
    enhanced_context = additional_context
    
    try:
        async with scraper:
            # Create tasks for scraping each URL
            scrape_tasks = []
            for url in urls:
                # Skip notion.so URLs - they require JavaScript rendering
                if 'notion.so' in url:
                    scraped_urls[url] = f"[Notion page: {url} - requires authentication]"
                    continue
                    
                # Parse the domain
                domain = urlparse(url).netloc
                
                # Check if this URL should be scraped
                if domain in scraped_urls:
                    logger.info(f"Skipping already scraped domain: {domain}")
                    continue
                    
                # Create a task for scraping
                scrape_tasks.append((url, scraper.scrape_page(url)))
            
            # Execute all scraping tasks concurrently with semaphore for rate limiting
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def scrape_with_semaphore(url, task):
                async with semaphore:
                    result = await task
                    return url, result
                    
            concurrent_tasks = [scrape_with_semaphore(url, task) for url, task in scrape_tasks]
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Failed to scrape: {str(result)}")
                    continue
                    
                url, content = result
                domain = urlparse(url).netloc
                
                if content is None:
                    logger.warning(f"No content scraped from {url}")
                    scraped_urls[url] = f"[No content available from {url}]"
                else:
                    # Extract relevant content
                    text = content.get('text', '')
                    if text:
                        # Truncate long content (limit to ~1000 words)
                        words = text.split()
                        if len(words) > 1000:
                            text = ' '.join(words[:1000]) + ' [truncated due to length]'
                        
                        # Store scraped content
                        scraped_urls[url] = text
            
            # Add scraped content to the context
            for url in urls:
                if url in scraped_urls:
                    # Replace URL in context with content or placeholder
                    content_marker = f"[Content from {url}]:\n{scraped_urls[url]}\n[End of content]"
                    enhanced_context = enhanced_context.replace(url, content_marker)
    
    except Exception as e:
        logger.error(f"Error during URL scraping: {str(e)}")
        return additional_context  # Return original context on error
    
    return enhanced_context

# Helper function to run async code
def run_async(coroutine):
    """Run an async function within a synchronous environment"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

# Usage examples
if __name__ == "__main__":
    # Example usage with redirect URL
    async def example():
        # Test with a redirect URL
        redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWQVqAIR1NgEvyQmYTsC52z8ia9UbM_xxEeYvs3FGar98jpSd3j-zE-aCMOiX2n3zC6F_WsTAtnCDHqcTAarum6LwTRSEidwSyVFyiSuVcVr0YbQ"
        
        scraper_config = ScraperConfig(
            respect_robots=True,
            request_delay=1.0,
            max_redirect_depth=5,
            follow_redirects=True
        )
        
        async with EthicalWebScraper(scraper_config) as scraper:
            # First try to resolve the redirect
            final_url = await scraper._resolve_redirects(redirect_url)
            print(f"Resolved redirect: {redirect_url} -> {final_url}")
            
            # Check if allowed by robots.txt
            print(f"Checking if {redirect_url} is allowed by robots.txt...")
            allowed = await scraper.check_robots_allowed(redirect_url)
            print(f"Is {redirect_url} allowed by robots.txt? {allowed}")
            
            if allowed:
                # Try to scrape the content
                content = await scraper.scrape_page(redirect_url)
                if content:
                    print(f"Successfully scraped content")
                    print(f"Title: {content.get('title', 'No title')}")
                    print(f"Content length: {len(content.get('text', ''))}")
                else:
                    print(f"Could not scrape content from {redirect_url}")
            else:
                print(f"Could not scrape {redirect_url} (disallowed by robots.txt)")
                
    # Run the example
    asyncio.run(example())