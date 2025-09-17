import streamlit as st
import json
from google import genai
from google.genai import types
import logging
import os
import sys
import asyncio
import aiohttp
import trafilatura
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
import time
from typing import Optional, Dict, List, Tuple
import hashlib
import re
from bs4 import BeautifulSoup
import socket
import random
from dataclasses import dataclass
import base64

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API credentials
GEMINI_API_KEY = st.secrets.get("GOOGLE_AI_STUDIO", "")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

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
    javascript_rendering: bool = False  # Enable JavaScript rendering (requires playwright)
    proxy: Optional[str] = None  # Proxy server URL

class EthicalWebScraper:
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config if config else ScraperConfig()
        self.domain_delays: Dict[str, float] = {}
        self.last_request_times: Dict[str, float] = {}
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Tuple[float, str]] = {}  # URL: (timestamp, content)
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.blacklist = self._load_blacklist()
        
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
    
    async def _get_robots_parser(self, url: str) -> Optional[RobotFileParser]:
        """Get or create robots.txt parser for domain"""
        if not self.config.respect_robots:
            return None
            
        domain = self._get_domain(url)
        if domain in self.robots_parsers:
            return self.robots_parsers[domain]
        
        robots_url = f"{urlparse(url).scheme}://{domain}/robots.txt"
        rp = RobotFileParser()
        
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
        except Exception as e:
            logger.warning(f"Could not fetch robots.txt for {domain}: {str(e)}")
            return None
        
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
            
        domain = self._get_domain(url)
        
        try:
            # Check cache first
            if self.config.cache_enabled and url in self.cache:
                timestamp, content = self.cache[url]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"Using cached content for {url}")
                    return content
                else:
                    del self.cache[url]
            
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
                
                if 'html' not in content_type.lower():
                    logger.warning(f"Non-HTML content type: {content_type}")
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
                    self.cache[url] = (time.time(), content)
                
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
    
    async def _extract_content(self, html: str, url: str) -> Optional[Dict]:
        """Extract structured content from HTML"""
        try:
            # First try with trafilatura (fast and good for articles)
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
                    soup = BeautifulSoup(html, 'lxml')
                    
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
                        result['source'] = 'BeautifulSoup fallback'
                
                # Add metadata
                result['url'] = url
                result['timestamp'] = time.time()
                result['content_hash'] = hashlib.sha256(result.get('text', '').encode()).hexdigest()
                
                return result
            return None
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return None
    
    async def scrape_page(self, url: str) -> Optional[Dict]:
        """Main method to scrape a single page"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        html = await self._fetch_url(url)
        if not html:
            return None
            
        return await self._extract_content(html, url)
    
    def extract_metadata(self, html: str) -> Dict:
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
            soup = BeautifulSoup(html, 'lxml')
            
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
    
    async def scrape_with_metadata(self, url: str) -> Optional[Dict]:
        """Scrape page including detailed metadata"""
        html = await self._fetch_url(url)
        if not html:
            return None
            
        content = await self._extract_content(html, url)
        if not content:
            return None
            
        metadata = await asyncio.to_thread(self.extract_metadata, html)
        content['metadata'] = metadata
        
        return content

# Helper function to run async code in Streamlit
def run_async(coroutine):
    """Run an async function within Streamlit's synchronous environment"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

def initialize_session_state():
    """Initialize session state variables."""
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None
    if "content_type" not in st.session_state:
        st.session_state["content_type"] = None
    if "input_url" not in st.session_state:
        st.session_state["input_url"] = ""
    if "enable_web_scraping" not in st.session_state:
        st.session_state["enable_web_scraping"] = True
    if "scraped_content" not in st.session_state:
        st.session_state["scraped_content"] = None
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = None
    if "manual_content" not in st.session_state:
        st.session_state["manual_content"] = ""
    if "robots_blocked" not in st.session_state:
        st.session_state["robots_blocked"] = False
    if "awaiting_manual_content" not in st.session_state:
        st.session_state["awaiting_manual_content"] = False
    if "proceed_without_scraping" not in st.session_state:
        st.session_state["proceed_without_scraping"] = False

async def extract_youtube_thumbnail(video_url: str) -> str:
    """Extract YouTube thumbnail URL from video URL"""
    try:
        video_id = None
        if "youtube.com/watch" in video_url and "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        
        if video_id:
            # Return maxres thumbnail if available, otherwise high quality
            return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    except Exception:
        pass
    return ""

async def validate_thumbnail(image_url: str, content_title: str) -> dict:
    """Validate if thumbnail is relevant using Gemini"""
    try:
        # Skip validation for YouTube thumbnails
        if "img.youtube.com" in image_url:
            return {
                "suitable": True,
                "reason": "YouTube official thumbnail",
                "alternative_suggestion": ""
            }

        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Download the image and convert to base64
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    mime_type = response.headers.get('Content-Type', 'image/jpeg')
                else:
                    return {
                        "suitable": False,
                        "reason": f"Could not download image (HTTP {response.status})",
                        "alternative_suggestion": ""
                    }

        prompt = f"""
        Analyze this image and determine if it would make a suitable thumbnail for content titled: "{content_title}".
        
        Return JSON with:
        - suitable (boolean): Whether the image is appropriate
        - reason: Brief explanation
        - alternative_suggestion: Alternative image description if not suitable
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_data(
                            mime_type=mime_type,
                            data=base64_image
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        if hasattr(response, 'candidates') and response.candidates:
            content = response.candidates[0].content
            if hasattr(content, 'parts') and content.parts:
                return json.loads(content.parts[0].text)
                
    except Exception as e:
        logger.error(f"Thumbnail validation error: {str(e)}")
    
    # Fallback if validation fails
    return {
        "suitable": True,  # Assume suitable if we can't validate
        "reason": "Validation failed, using thumbnail anyway",
        "alternative_suggestion": ""
    }


async def extract_article_thumbnail(url: str, html: str = None) -> str:
    """Extract article thumbnail from URL or HTML"""
    try:
        # If HTML is provided, parse it directly
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            # Check Open Graph image first
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return og_image['content']
            
            # Check Twitter card image
            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                return twitter_image['content']
            
            # Fallback to first large image in article
            for img in soup.find_all('img'):
                if img.get('src') and ('logo' not in img.get('src', '').lower()):
                    return img['src']
        
        # If no HTML, try to fetch just enough to get metadata
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return await extract_article_thumbnail(url, content)
    except Exception:
        pass
    return ""

async def extract_basic_info(url: str, content_type: str) -> dict:
    """
    Extract basic information from URL to improve search relevance
    
    Args:
        url: The URL to analyze
        content_type: Either "video" or "article"
    
    Returns:
        Basic information including keywords for better searching
    """
    try:
        # For YouTube videos, try to extract video ID and title from the URL
        if content_type == "video" and ("youtube.com" in url or "youtu.be" in url):
            video_id = None
            
            # Extract video ID from URL
            if "youtube.com/watch" in url and "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
                
            if video_id:
                # Use a simple HTTP request to get the page title (which contains the video title)
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(f"https://www.youtube.com/watch?v={video_id}") as response:
                            if response.status == 200:
                                html = await response.text()
                                
                                # Extract title using regex
                                import re
                                title_match = re.search(r'<title>(.*?)</title>', html)
                                if title_match:
                                    title = title_match.group(1)
                                    # Clean up title (remove " - YouTube" suffix)
                                    if " - YouTube" in title:
                                        title = title.replace(" - YouTube", "")
                                        
                                    # Create search keywords from the title
                                    return {
                                        "title": title,
                                        "video_id": video_id,
                                        "search_keywords": f"{title} AI technology information guide"
                                    }
                    except Exception as e:
                        logger.warning(f"Error fetching video title: {str(e)}")
            
        # For articles, try to extract domain and path
        elif content_type == "article":
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            
            # Try to create meaningful keywords from the URL path
            if path:
                # Convert path to keywords
                path_parts = path.strip('/').split('/')
                keywords = ' '.join([part.replace('-', ' ').replace('_', ' ') for part in path_parts if part])
                
                if keywords:
                    return {
                        "domain": domain,
                        "path": path,
                        "search_keywords": f"{keywords} {domain} information"
                    }
        
        # Default fallback
        return {
            "search_keywords": f"{url} information guide"
        }
    
    except Exception as e:
        logger.error(f"Error extracting basic info: {str(e)}")
        return {
            "search_keywords": url
        }

async def search_and_analyze(url: str, content_type: str) -> dict:
    """
    Use Google Search to provide additional information for content analysis.
    """
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={
                "base_url": 'https://gateway.helicone.ai',
                "headers": {
                    "helicone-auth": f'Bearer {st.secrets["HELICONE_API_KEY"]}',
                    "helicone-target-url": 'https://generativelanguage.googleapis.com'
                }
            }
        )
        model = "gemini-2.0-flash"
        
        # First, extract basic information about the content to create a more accurate search query
        basic_info = await extract_basic_info(url, content_type)
        search_keywords = basic_info.get('search_keywords', url)
        
        # Create prompt based on content type and extracted keywords
        if content_type == "video":
            search_prompt = f"""
            Search for information about: {search_keywords}
            
            Analyze this YouTube video and provide comprehensive JSON output including:
            1. Metadata (title, channel, publication date, view count, duration)
            2. Detailed summary of the video content
            3. Key insights with relevant timestamps
            4. Key topics discussed with quotes
            5. Economic context and implications
            6. Technical details if applicable
            
            Format the response as JSON with the following structure:
            {{
                "metadata": {{
                    "title": "Video title",
                    "channel": "Channel name",
                    "date": "Publication date",
                    "views": "Number of views",
                    "duration": "Video duration",
                    "url": "Video URL",
                    "thumbnail": "Thumbnail URL"
                }},
                "contentTags": ["list", "of", "descriptive", "tags"],
                "summary": "Detailed summary of video content",
                "keyInsights": [
                    {{
                        "insight": "Key insight or point",
                        "timestamp": "Approximate timestamp in video",
                        "significance": "Why this point matters"
                    }}
                ],
                "keyTopics": [
                    {{
                        "name": "Topic name",
                        "description": "Topic description",
                        "quotes": [
                            {{
                                "text": "Relevant quote",
                                "timestamp": "Quote timestamp"
                            }}
                        ]
                    }}
                ],
                "economicContext": {{
                    "implications": "Potential implications or significance",
                    "marketImpact": {{
                        "aiSector": "Impact on AI sector",
                        "developerTools": "Impact on developer tools"
                    }}
                }},
                "technicalDetails": {{
                    "benchmarkImprovements": {{
                        "math": "Math benchmark improvement",
                        "science": "Science benchmark improvement"
                    }},
                    "newFeatures": ["list", "of", "new", "features"]
                }}
            }}
            """
        else:  # article
            search_prompt = f"""
            Search for information about: {search_keywords}
            
            Analyze this article and provide comprehensive JSON output including:
            1. Metadata (title, source, author, publication date)
            2. Detailed summary
            3. Key points
            4. Expert opinions
            5. Technical analysis
            6. Developer features if applicable
            
            Format the response as JSON with the following structure:
            {{
                "metadata": {{
                    "title": "Article title",
                    "source": "Source name",
                    "author": "Author name",
                    "date": "Publication date",
                    "readingTime": "Estimated reading time",
                    "url": "Article URL",
                    "thumbnail": "Thumbnail URL"
                }},
                "contentTags": ["list", "of", "descriptive", "tags"],
                "summary": "Detailed summary of article content",
                "keyPoints": ["list", "of", "key", "points"],
                "expertOpinions": [
                    {{
                        "name": "Expert name",
                        "title": "Expert title",
                        "quote": "Expert quote"
                    }}
                ],
                "technicalAnalysis": {{
                    "architectureChanges": ["list", "of", "architecture", "changes"],
                    "performanceMetrics": {{
                        "mathBenchmark": "Math benchmark score",
                        "scienceBenchmark": "Science benchmark score",
                        "reasoningTests": "Reasoning test score"
                    }}
                }},
                "developerFeatures": [
                    {{
                        "name": "Feature name",
                        "description": "Feature description",
                        "example": "Usage example"
                    }}
                ]
            }}
            """

        
        # Set up the initial conversation flow
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"{search_keywords}\n\nFind additional information and context"),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text="I'll search for additional information about this content."),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=search_prompt),
                ],
            ),
        ]

        
        tools = [
            types.Tool(google_search=types.GoogleSearch())
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=tools,
            response_mime_type="application/json",
        )
        
        st.info("Searching for additional context and related content...")
        
        # Use non-streaming response to get JSON
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        if hasattr(response, 'candidates') and response.candidates:
            content = response.candidates[0].content
            if hasattr(content, 'parts') and content.parts:
                text_response = content.parts[0].text
                
                # Try to parse the JSON response
                try:
                    # Find the JSON part if it's embedded in text
                    json_start = text_response.find('{')
                    json_end = text_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = text_response[json_start:json_end]
                        search_results = json.loads(json_str)
                    else:
                        search_results = json.loads(text_response)
                    
                    return {
                        "success": True,
                        "data": search_results
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                    return {
                        "success": False,
                        "error": "Could not parse JSON from search results",
                        "raw_response": text_response
                    }
        
        return {
            "success": False,
            "error": "No valid response received from the search API",
            "raw_response": str(response)
        }
    
    except Exception as e:
        logger.error(f"Error in search analysis: {str(e)}")
        return {
            "success": False,
            "error": f"An error occurred during search: {str(e)}"
        }

async def analyze_content(url: str, content_type: str, enable_scraping: bool = True, manual_content: str = "", proceed_without_scraping: bool = False) -> dict:
    """Enhanced content analysis with better error handling"""
    try:
        # Initialize thumbnail variables
        thumbnail_url = ""
        thumbnail_validation = {}
        
        # Get thumbnail based on content type
        if content_type == "video":
            thumbnail_url = await extract_youtube_thumbnail(url)
            thumbnail_validation = {
                "suitable": True,
                "reason": "YouTube official thumbnail",
                "alternative_suggestion": ""
            }
        else:
            if enable_scraping and not proceed_without_scraping:
                async with EthicalWebScraper() as scraper:
                    html = await scraper._fetch_url(url)
                    thumbnail_url = await extract_article_thumbnail(url, html)
                    if thumbnail_url and "img.youtube.com" not in thumbnail_url:
                        thumbnail_validation = await validate_thumbnail(thumbnail_url, url)
                        if not thumbnail_validation.get('suitable', False):
                            thumbnail_url = ""

        # Initialize client with better error handling
        try:
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={
                    "base_url": 'https://gateway.helicone.ai',
                    "headers": {
                        "helicone-auth": f'Bearer {st.secrets["HELICONE_API_KEY"]}',
                        "helicone-target-url": 'https://generativelanguage.googleapis.com'
                    }
                }
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize API client: {str(e)}"
            }

        model = "gemini-2.0-flash"
        
        # Initialize web scraper if needed
        scraper = None
        scraped_content = None
        if content_type == "article":
            if manual_content and manual_content.strip():
                scraped_content = {
                    "text": manual_content.strip(),
                    "metadata": {
                        "title": "Manually provided content",
                        "source": url,
                        "authors": [],
                        "published_date": "",
                    },
                    "url": url,
                    "timestamp": time.time(),
                    "content_hash": hashlib.sha256(manual_content.strip().encode()).hexdigest()
                }
            elif enable_scraping and not proceed_without_scraping:
                scraper_config = ScraperConfig(
                    request_delay=2.0,
                    user_agent="ContentAnalysisTool/1.0 (+https://github.com/ethical-content-analysis)"
                )
                scraper = EthicalWebScraper(scraper_config)
                try:
                    async with scraper:
                        scraped_content = await scraper.scrape_with_metadata(url)
                except Exception as e:
                    logger.error(f"Scraping error: {str(e)}")

        # Create prompt based on content type
        prompt = f"""
        Analyze this {'video' if content_type == 'video' else 'article'} and provide comprehensive JSON output.
        Include metadata, summary, key points, and relevant analysis.
        
        Then provide specific recommendations to "Make It Useful" by:
        1. Identifying the target audience
        2. Suggesting practical applications
        3. Providing actionable takeaways
        4. Offering implementation tips

        
        {'Video URL' if content_type == 'video' else 'Article URL'}: {url}
        """
        
        if scraped_content:
            prompt += f"\n\nContent:\n{scraped_content.get('text', '')}"
        elif manual_content:
            prompt += f"\n\nContent:\n{manual_content}"

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        # Configure for extended output
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=20000,
            response_mime_type="application/json",
        )

        # Execute the analysis with timeout
        try:
            # Use a regular call since client.models.generate_content is not a coroutine
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            # Only add timeout for network operations if needed
            # You could wrap this in asyncio.to_thread if you want it to be non-blocking
        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }

        # Process response with robust error handling
        if not hasattr(response, 'candidates') or not response.candidates:
            return {
                "success": False,
                "error": "No candidates in API response",
                "raw_response": str(response)
            }
            
        content = response.candidates[0].content
        if not hasattr(content, 'parts') or not content.parts:
            return {
                "success": False,
                "error": "No content parts in response",
                "raw_response": str(response)
            }
            
        text_response = content.parts[0].text
        
        # Try to parse response as JSON
        try:
            # Handle case where JSON might be embedded in text
            json_str = text_response
            json_start = text_response.find('{')
            if json_start >= 0:
                json_end = text_response.rfind('}') + 1
                json_str = text_response[json_start:json_end]
            
            result = json.loads(json_str)
            
            # Validate basic structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a JSON object")
                
            # Add thumbnail info if available
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata']['thumbnail'] = thumbnail_url
            if thumbnail_validation:
                result['thumbnail_validation'] = thumbnail_validation
                
            return {
                "success": True,
                "data": result,
                "content_type": content_type,
                "scraped_content_used": scraped_content is not None,
                "manually_provided": bool(manual_content and manual_content.strip())
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Could not parse JSON response: {str(e)}",
                "raw_response": text_response
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing response: {str(e)}",
                "raw_response": text_response
            }
            
    except Exception as e:
        logger.error(f"Error in content analysis: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"An error occurred during analysis: {str(e)}"
        }

def transform_video_analysis(raw_data: dict) -> dict:
    """Transform raw video analysis to match Streamlit expected structure"""
    result = {
        "metadata": {
            "title": raw_data.get("title", ""),
            "channel": raw_data.get("channel", ""),
            "date": raw_data.get("date", ""),
            "duration": raw_data.get("duration", ""),
            "views": raw_data.get("views", ""),
            "url": raw_data.get("url", ""),
            "thumbnail": raw_data.get("thumbnail", "")
        },
        "contentTags": raw_data.get("contentTags", []),
        "summary": raw_data.get("summary", ""),
        "keyInsights": [],
        "keyTopics": [],
        "economicContext": {
            "implications": raw_data.get("economicContext", ""),
            "marketImpact": {
                "aiSector": "",
                "developerTools": ""
            }
        },
        "technicalDetails": {
            "benchmarkImprovements": {
                "math": "",
                "science": ""
            },
            "newFeatures": []
        }
    }
    
    # Transform key insights
    if "keyInsights" in raw_data:
        if isinstance(raw_data["keyInsights"], list):
            result["keyInsights"] = [
                {
                    "insight": insight.get("insight", ""),
                    "timestamp": insight.get("timestamp", ""),
                    "significance": insight.get("significance", "")
                }
                for insight in raw_data["keyInsights"]
            ]
    
    # Transform key topics
    if "keyTopics" in raw_data:
        if isinstance(raw_data["keyTopics"], list):
            result["keyTopics"] = [
                {
                    "name": topic.get("name", ""),
                    "description": topic.get("description", ""),
                    "quotes": [
                        {
                            "text": quote.get("text", ""),
                            "timestamp": quote.get("timestamp", "")
                        }
                        for quote in topic.get("quotes", [])
                    ]
                }
                for topic in raw_data["keyTopics"]
            ]
    
    # Add technical details if available
    if "technicalDetails" in raw_data:
        result["technicalDetails"] = raw_data["technicalDetails"]
    
    return result

def transform_article_analysis(raw_data: dict) -> dict:
    """Transform raw article analysis to match Streamlit expected structure"""
    result = {
        "metadata": {
            "title": raw_data.get("title", ""),
            "source": raw_data.get("source", ""),
            "author": raw_data.get("author", ""),
            "date": raw_data.get("date", ""),
            "readingTime": raw_data.get("readingTime", ""),
            "url": raw_data.get("url", ""),
            "thumbnail": raw_data.get("thumbnail", "")
        },
        "contentTags": raw_data.get("contentTags", []),
        "summary": raw_data.get("summary", ""),
        "keyPoints": raw_data.get("keyPoints", []),
        "expertOpinions": [],
        "technicalAnalysis": {
            "architectureChanges": [],
            "performanceMetrics": {
                "mathBenchmark": "",
                "scienceBenchmark": "",
                "reasoningTests": ""
            }
        },
        "developerFeatures": []
    }
    
    # Transform expert opinions
    if "expertOpinions" in raw_data:
        if isinstance(raw_data["expertOpinions"], list):
            result["expertOpinions"] = [
                {
                    "name": opinion.get("name", ""),
                    "title": opinion.get("title", ""),
                    "quote": opinion.get("quote", "")
                }
                for opinion in raw_data["expertOpinions"]
            ]
    
    # Add technical analysis if available
    if "technicalAnalysis" in raw_data:
        result["technicalAnalysis"] = raw_data["technicalAnalysis"]
    
    # Add developer features if available
    if "developerFeatures" in raw_data:
        result["developerFeatures"] = raw_data["developerFeatures"]
    
    return result

def display_video_analysis(analysis):
    """Display video analysis results with clickable thumbnail."""
    if isinstance(analysis, list):
        analysis = analysis[0]  # Take the first item if it's a list

    # Extract metadata with fallbacks
    metadata = analysis.get('metadata', {})
    title = metadata.get('title', 'Unknown Title')
    channel = metadata.get('author', 'Unknown')
    date = metadata.get('upload_date', 'Unknown')
    duration = metadata.get('duration', 'Unknown')
    views = metadata.get('views', 'Unknown')
    video_url = metadata.get('url', '')
    thumbnail_url = metadata.get('thumbnail', '')

    st.header(f"🎬 Video Analysis: {title}")
    
    # Display tags if available
    if 'tags' in metadata:
        st.subheader("📌 Key Topics")
        display_tags(metadata['tags'])
    
    # Basic info in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Channel</p>
            <p style="margin: 0; color: #4B5563;">{channel}</p>
        </div>
        <div style="margin-bottom: 1rem;">
            <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Published Date</p>
            <p style="margin: 0; color: #4B5563;">{date}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Duration</p>
            <p style="margin: 0; color: #4B5563;">{duration}</p>
        </div>
        <div style="margin-bottom: 1rem;">
            <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Views</p>
            <p style="margin: 0; color: #4B5563;">{views}</p>
        </div>
        """, unsafe_allow_html=True)

    # Display clickable thumbnail if available
    if thumbnail_url:
        st.subheader("Video Thumbnail")
        if video_url:
            st.markdown(
                f"""<a href="{video_url}" target="_blank">
                <img src="{thumbnail_url}" style="max-height: 300px; border-radius: 7px; cursor: pointer; display: block; margin: 0 auto 20px auto;">
                </a>""",
                unsafe_allow_html=True,
            )
        else:
            st.image(thumbnail_url, caption="Video Thumbnail", use_column_width=True)
            
    # Summary
    st.subheader("Summary")
    st.markdown(f"""
    <div style="background-color: #F9FAFB; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6;">
        {analysis.get('summary', 'No summary available.')}
    </div>
    """, unsafe_allow_html=True)
    
    # Key Points - Using the key_points from your data structure
    if 'key_points' in analysis:
        st.subheader("Key Points")
        for point in analysis['key_points']:
            with st.expander(f"📌 {point.get('point', 'Untitled')[:50]}..."):
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Point</p>
                    <p style="margin: 0; color: #4B5563;">{point.get('point', 'N/A')}</p>
                </div>
                <div>
                    <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Timestamp</p>
                    <p style="margin: 0; color: #4B5563;">{point.get('timestamp', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Analysis section if available
    if 'analysis' in analysis:
        st.subheader("Analysis")
        
        if 'overall_sentiment' in analysis['analysis']:
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Overall Sentiment</p>
                <p style="margin: 0; color: #4B5563;">{analysis['analysis']['overall_sentiment']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'strengths' in analysis['analysis']:
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Strengths</p>
            </div>
            """, unsafe_allow_html=True)
            for strength in analysis['analysis']['strengths']:
                st.markdown(f"- {strength}")
        
        if 'weaknesses' in analysis['analysis']:
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Weaknesses</p>
            </div>
            """, unsafe_allow_html=True)
            for weakness in analysis['analysis']['weaknesses']:
                st.markdown(f"- {weakness}")
        
        if 'target_audience' in analysis['analysis']:
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Target Audience</p>
            </div>
            """, unsafe_allow_html=True)
            for audience in analysis['analysis']['target_audience']:
                st.markdown(f"- {audience}")
        
        if 'potential_biases' in analysis['analysis']:
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Potential Biases</p>
            </div>
            """, unsafe_allow_html=True)
            for bias in analysis['analysis']['potential_biases']:
                st.markdown(f"- {bias}")

    # Add "Make It Useful" section
    if 'make_it_useful' in analysis:
        st.subheader("✨ Make It Useful")
        useful = analysis['make_it_useful']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #F0FDF4; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="color: #166534; margin-top: 0;">Target Audience</h4>
                <p>{}</p>
            </div>
            """.format(useful.get('target_audience', 'Not specified')), unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: #EFF6FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="color: #1E40AF; margin-top: 0;">Practical Applications</h4>
                <ul style="margin-bottom: 0;">
                    {}
                </ul>
            </div>
            """.format(''.join([f'<li>{item}</li>' for item in useful.get('practical_applications', [])])), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #FEF2F2; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="color: #991B1B; margin-top: 0;">Actionable Takeaways</h4>
                <ul style="margin-bottom: 0;">
                    {}
                </ul>
            </div>
            """.format(''.join([f'<li>{item}</li>' for item in useful.get('actionable_takeaways', [])])), unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: #F5F3FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="color: #5B21B6; margin-top: 0;">Implementation Tips</h4>
                <ul style="margin-bottom: 0;">
                    {}
                </ul>
            </div>
            """.format(''.join([f'<li>{item}</li>' for item in useful.get('implementation_tips', [])])), unsafe_allow_html=True)
    
                
def display_search_results(results):
    """Display multiple search results with clickable thumbnails."""
    if not results or not isinstance(results, list):
        return
    
    # Create lists for clickable_images parameters
    thumbnails = []
    titles = []
    urls = []
    
    # Collect data for each result
    for result in results:
        thumbnail = result.get('thumbnail', '')
        title = result.get('title', 'Untitled')
        url = result.get('url', '')
        
        if thumbnail:  # Only add items with thumbnails
            thumbnails.append(thumbnail)
            titles.append(title)
            urls.append(url)
    
    if thumbnails:
        st.subheader("Related Content")
        clicked = clickable_images(
            thumbnails,
            titles=titles,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "10px", "height": "150px", "border-radius": "5px", "cursor": "pointer"}
        )
        
        # Handle click event
        if clicked >= 0 and clicked < len(urls):
            clicked_url = urls[clicked]
            st.markdown(f"[Open {titles[clicked]} in New Tab]({clicked_url})")
            
def display_tags(tags, container=None):
    """Display tags as colored pills matching the financial dashboard style."""
    if not tags:
        return
        
    if not container:
        container = st
        
    if isinstance(tags, str):
        tags = [tags]
    elif isinstance(tags, dict):
        tags = list(tags.keys())
    
    # Create HTML for tags
    tag_html = '<div class="tag-container">'
    for tag in tags:
        if not tag:
            continue
            
        # Determine tag color class based on content
        if "Fed" in tag or "Interest" in tag:
            tag_class = "blue-tag"
        elif "Economy" in tag or "Market" in tag:
            tag_class = "green-tag"
        elif "Analysis" in tag:
            tag_class = "purple-tag"
        else:
            tag_class = "yellow-tag"
            
        tag_html += f'<span class="tag {tag_class}">{tag}</span>'
    
    tag_html += '</div>'
    
    # Display the tags
    container.markdown(tag_html, unsafe_allow_html=True)


def display_article_analysis(analysis):
    """Display article analysis results with proper formatting."""
    if isinstance(analysis, list):
        analysis = analysis[0]  # Take the first item if it's a list

    # Extract metadata with fallbacks
    metadata = analysis.get('metadata', {})
    title = metadata.get('title', 'Unknown Title')
    source = metadata.get('source', 'Unknown Source')
    url = metadata.get('url', '')
    description = metadata.get('description', '')
    thumbnail_url = metadata.get('thumbnail', '')

    st.header(f"📰 Article Analysis: {title}")
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Source</p>
            <p style="margin: 0; color: #4B5563;">{source}</p>
        </div>
        <div style="margin-bottom: 1rem;">
            <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Description</p>
            <p style="margin: 0; color: #4B5563;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if url:
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">URL</p>
                <p style="margin: 0; color: #4B5563;"><a href="{url}" target="_blank">View Original</a></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary
    st.subheader("Summary")
    st.markdown(f"""
    <div style="background-color: #F9FAFB; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6;">
        {analysis.get('summary', 'No summary available.')}
    </div>
    """, unsafe_allow_html=True)
    
    # Key Points
    if 'key_points' in analysis and isinstance(analysis['key_points'], list):
        st.subheader("Key Points")
        for i, point in enumerate(analysis['key_points'], 1):
            st.markdown(f"""
            <div style="background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <div style="display: flex; align-items: flex-start;">
                    <div style="background-color: #3B82F6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem; flex-shrink: 0;">{i}</div>
                    <div>
                        <p style="margin: 0; color: #1F2937;">{point}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis section if available
    if 'analysis' in analysis:
        analysis_data = analysis['analysis']
        st.subheader("Analysis")
        
        if 'scope' in analysis_data:
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Scope</p>
                <p style="margin: 0; color: #4B5563;">{analysis_data['scope']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'intended_audience' in analysis_data:
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Intended Audience</p>
                <p style="margin: 0; color: #4B5563;">{analysis_data['intended_audience']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'potential_use_cases' in analysis_data:
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Potential Use Cases</p>
                <p style="margin: 0; color: #4B5563;">{analysis_data['potential_use_cases']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'strengths' in analysis_data:
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Strengths</p>
            </div>
            """, unsafe_allow_html=True)
            st.write(analysis_data['strengths'])
        
        if 'weaknesses' in analysis_data:
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <p style="font-weight: 600; margin-bottom: 0.25rem; color: #1F2937;">Weaknesses</p>
            </div>
            """, unsafe_allow_html=True)
            st.write(analysis_data['weaknesses'])
    
    # Make It Useful section
    if 'make_it_useful' in analysis:
        useful = analysis['make_it_useful']
        st.subheader("✨ Make It Useful")
        
        # Target Audience (single card)
        st.markdown(f"""
        <div style="background-color: #F0FDF4; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h4 style="color: #166534; margin-top: 0;">👥 Who Should Read This</h4>
            <p>{useful.get('target_audience', 'General audience')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Practical Applications (with bold handling)
        applications_html = "".join([
            f"<li>{app.replace('**', '').replace('**', '')}</li>" 
            for app in useful.get('practical_applications', ['No practical applications listed'])
        ])
        st.markdown(f"""
        <div style="background-color: #EFF6FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h4 style="color: #1E40AF; margin-top: 0;">🛠️ Practical Applications</h4>
            <ul style="margin-bottom: 0; padding-left: 1.5rem;">
                {applications_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Actionable Takeaways (with bold handling)
        takeaways_html = "".join([
            f"<li>{takeaway.replace('**', '').replace('**', '')}</li>" 
            for takeaway in useful.get('actionable_takeaways', ['No actionable takeaways listed'])
        ])
        st.markdown(f"""
        <div style="background-color: #FEF2F2; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h4 style="color: #991B1B; margin-top: 0;">✅ Actionable Takeaways</h4>
            <ul style="margin-bottom: 0; padding-left: 1.5rem;">
                {takeaways_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Implementation Tips (with bold handling)
        tips_html = "".join([
            f"<li>{tip.replace('**', '').replace('**', '')}</li>" 
            for tip in useful.get('implementation_tips', ['No implementation tips listed'])
        ])
        st.markdown(f"""
        <div style="background-color: #F5F3FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h4 style="color: #5B21B6; margin-top: 0;">💡 Implementation Tips</h4>
            <ul style="margin-bottom: 0; padding-left: 1.5rem;">
                {tips_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)



def display_results(results):
    """Enhanced results display with better error handling"""
    if not results:
        st.error("No results to display")
        return
        
    if results.get("success"):
        st.success("Analysis completed successfully!")
        
        # Show content type and source
        content_type = results.get("content_type", "content").capitalize()
        st.markdown(f"### {content_type} Analysis Results")
        
        # Display appropriate analysis
        if results["content_type"] == "video":
            display_video_analysis(results["data"])
        else:
            display_article_analysis(results["data"])
        
        # Show raw JSON in expander
        with st.expander("View Raw JSON Data"):
            st.json(results["data"])
            
        # Add download button
        json_str = json.dumps(results["data"], indent=2)
        st.download_button(
            label="Download Analysis",
            data=json_str,
            file_name=f"{results['content_type']}_analysis.json",
            mime="application/json"
        )
    else:
        st.error("Analysis failed")
        
        # Display detailed error information
        if "error" in results:
            st.error(f"Error: {results['error']}")
            
        if "raw_response" in results:
            with st.expander("Technical Details"):
                st.text(results["raw_response"])
        
        # Add troubleshooting tips
        st.markdown("""
        **Troubleshooting Tips:**
        - Check your internet connection
        - Verify your API keys are valid
        - Try a different URL
        - The content might be too long - try shorter content
        """)
        
        if st.button("Try Again"):
            st.session_state["analysis_results"] = None
            st.experimental_rerun()

def display_search_based_article_analysis(analysis, search_data):
    """Display article analysis based primarily on search results when no content is available."""
    # Handle case where analysis is a list
    if isinstance(analysis, list):
        analysis = analysis[0]  # Take the first item if it's a list
    
    st.header(f"📰 Article Analysis (Based on Search Results)")
    
    # Basic metadata from title
    st.subheader("Article Information")
    st.write(f"**Title:** {analysis.get('title', 'Unknown Title')}")
    
    # Display search-based context prominently
    st.subheader("Related Information")
    
    # Related Articles from search
    if 'relatedArticles' in search_data:
        st.write("**Related Articles:**")
        for article in search_data['relatedArticles']:
            st.write(f"- [{article.get('title', 'Untitled')}]({article.get('url', '#')}) - {article.get('source', 'Unknown source')}")
            st.write(f"  *{article.get('relevance', '')}*")
    
    # Additional Context from search
    if 'additionalContext' in search_data:
        st.write("**Additional Context:**")
        
        if 'recentDevelopments' in search_data['additionalContext']:
            st.write("*Recent Developments:*")
            for development in search_data['additionalContext']['recentDevelopments']:
                st.write(f"- {development}")
        
        if 'historicalContext' in search_data['additionalContext']:
            st.write("*Historical Context:*")
            st.write(search_data['additionalContext']['historicalContext'])
        
        if 'expertPerspectives' in search_data['additionalContext']:
            st.write("*Expert Perspectives:*")
            for perspective in search_data['additionalContext']['expertPerspectives']:
                st.write(f"- {perspective}")
    
    # Limited analysis if available
    if 'summary' in analysis and analysis['summary'] != 'No summary available.':
        st.subheader("Limited Analysis")
        st.write(analysis['summary'])

async def check_url_scrapeability(url: str) -> dict:
    """
    Check if a URL can be scraped according to robots.txt rules
    
    Args:
        url: The URL to check
    
    Returns:
        Dictionary with scrapeability status and related information
    """
    try:
        # Initialize minimal scraper just for robots.txt check
        scraper_config = ScraperConfig(
            request_delay=1.0,
            respect_robots=True
        )
        scraper = EthicalWebScraper(scraper_config)
        
        async with scraper:
            domain = urlparse(url).netloc
            robots_parser = await scraper._get_robots_parser(url)
            
            if robots_parser and not robots_parser.can_fetch(scraper.config.user_agent, url):
                logger.warning(f"URL {url} disallowed by robots.txt")
                return {
                    "scrapeable": False,
                    "message": f"The website {domain} disallows automated content extraction according to robots.txt"
                }
            else:
                return {
                    "scrapeable": True,
                    "message": f"The website {domain} allows scraping this URL"
                }
    except Exception as e:
        logger.error(f"Error checking robots.txt: {str(e)}")
        return {
            "scrapeable": False,
            "message": f"Error checking robots.txt: {str(e)}"
        }

def main():
    """Main function with proper async cleanup"""
    initialize_session_state()
    
    
    try:
        # Initialize asyncio loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        st.title("📹📰 Content Analysis Tool")
        st.markdown("Analyze YouTube videos and articles with detailed structured outputs.")
        
        with st.form("content_analysis_form"):
            url = st.text_input("Enter URL:", value=st.session_state.get("input_url", ""))
            content_type = st.radio("Content Type:", ["Video", "Article"])
            
            if content_type == "Article":
                enable_scraping = st.checkbox(
                    "Enable Ethical Web Scraping", 
                    value=st.session_state.get("enable_web_scraping", True),
                    disabled=st.session_state.get("robots_blocked", False)
                )
            else:
                enable_scraping = False
            
            submitted = st.form_submit_button("Analyze Content")
            
            if submitted:
                st.session_state["input_url"] = url
                st.session_state["content_type"] = content_type.lower()
                st.session_state["enable_web_scraping"] = enable_scraping
                
                if not url:
                    st.error("Please enter a URL")
                    return
                    
                with st.spinner(f"Analyzing {content_type} content..."):
                    try:
                        analysis = loop.run_until_complete(analyze_content(
                            url,
                            content_type.lower(),
                            enable_scraping,
                            st.session_state.get("manual_content", "")
                        ))
                        
                        if isinstance(analysis, dict) and analysis.get("error") == "robots_blocked":
                            st.session_state["robots_blocked"] = True
                        else:
                            st.session_state["analysis_results"] = analysis
                            st.session_state["robots_blocked"] = False
                            
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        st.session_state["analysis_results"] = {
                            "success": False,
                            "error": str(e)
                        }
        
        # Show manual input section if robots.txt blocked scraping
        if st.session_state.get("robots_blocked"):
            st.warning("⚠️ Robots.txt Restriction Detected")
            with st.form("manual_content_form"):
                manual_content = st.text_area("Article Content (Optional)", height=300)
                submit_manual = st.form_submit_button("Analyze with Manual Content")
                submit_without = st.form_submit_button("Continue with URL Only")
                
                if submit_manual or submit_without:
                    st.session_state["manual_content"] = manual_content if submit_manual else ""
                    with st.spinner("Analyzing content..."):
                        try:
                            analysis = loop.run_until_complete(analyze_content(
                                st.session_state["input_url"],
                                st.session_state["content_type"],
                                False,
                                st.session_state["manual_content"] if submit_manual else "",
                                True
                            ))
                            st.session_state["analysis_results"] = analysis
                        except Exception as e:
                            st.error(f"Analysis error: {str(e)}")
        
        # Display results if available
        if st.session_state.get("analysis_results"):
            display_results(st.session_state["analysis_results"])
            
    finally:
        # Clean up
        try:
            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()




if __name__ == "__main__":
    # Add custom CSS
    st.markdown("""
    <style>
        /* Improved insight styling */
        .insight-card {
            background-color: #F9FAFB;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #3B82F6;
        }
        .insight-number {
            background-color: #3B82F6;
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.75rem;
            flex-shrink: 0;
        }
        .insight-header {
            font-weight: 600;
            margin-bottom: 0.25rem;
            color: #1F2937;
        }
        .insight-content {
            margin: 0;
            color: #4B5563;
        }

        /* Tag styling from financial dashboard */
        .tag-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }
        .tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
            margin-top: 0.5rem;
        }
        .blue-tag {
            background-color: #EFF6FF;
            color: #1E40AF;
        }
        .green-tag {
            background-color: #ECFDF5;
            color: #065F46;
        }
        .purple-tag {
            background-color: #F5F3FF;
            color: #5B21B6;
        }
        .yellow-tag {
            background-color: #FFFBEB;
            color: #92400E;
        }

        /* Make It Useful section styles */
        .useful-section {
            background-color: #F8FAFC;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-top: 1.5rem;
            border: 1px solid #E2E8F0;
        }
        .useful-header {
            color: #3B82F6;
            font-size: 1.25rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        .useful-header svg {
            margin-right: 0.5rem;
        }
        .useful-card {
            background-color: white;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .useful-card-title {
            font-weight: 600;
            color: #1E40AF;
            margin-bottom: 0.5rem;
        }

        
    </style>
    """, unsafe_allow_html=True)
    
    main()