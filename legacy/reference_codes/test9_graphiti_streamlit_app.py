"""
Graphiti Knowledge Graph Visualization with Streamlit
Using Streamlit's proper async pattern with fixed Neo4j connection handling
"""

import streamlit as st
import asyncio
import json
import os
from datetime import datetime, timezone
from pyvis.network import Network
import nest_asyncio
import uuid
import traceback
from functools import wraps

# Apply nest_asyncio to patch the event loop
nest_asyncio.apply()

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

# Page configuration
st.set_page_config(page_title="Graphiti Explorer", page_icon="🔍", layout="wide")

# Initialize session state
if "graph_initialized" not in st.session_state:
    st.session_state.graph_initialized = False
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "node_search_results" not in st.session_state:
    st.session_state.node_search_results = []
if "episodes_added" not in st.session_state:
    st.session_state.episodes_added = False
if "visualization_data" not in st.session_state:
    st.session_state.visualization_data = {"nodes": [], "edges": []}
if "status_message" not in st.session_state:
    st.session_state.status_message = ""
if "reranked_results" not in st.session_state:
    st.session_state.reranked_results = []

# Neo4j connection parameters
if 'neo4j' in st.secrets:
    neo4j_uri = st.secrets.neo4j.NEO4J_URI
    neo4j_user = st.secrets.neo4j.NEO4J_USERNAME
    neo4j_password = st.secrets.neo4j.NEO4J_PASSWORD
else:
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

# Google Gemini API key
if 'GEMINI_API_KEY' in st.secrets:
    gemini_api_key = st.secrets.GEMINI_API_KEY
else:
    gemini_api_key = os.environ.get('GEMINI_API_KEY')

# Check if credentials are available
credentials_available = all([neo4j_uri, neo4j_user, neo4j_password, gemini_api_key])
if not credentials_available:
    st.error("Missing credentials. Please ensure Neo4j and Google Gemini credentials are set.")
    st.stop()

# Sample episodes
sample_episodes = [
    {
        'content': 'Kamala Harris is the Attorney General of California. She was previously the district attorney for San Francisco.',
        'type': EpisodeType.text,
        'description': 'political biography',
    },
    {
        'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
        'type': EpisodeType.text,
        'description': 'political timeline',
    },
    {
        'content': {
            'name': 'Gavin Newsom',
            'position': 'Governor',
            'state': 'California',
            'previous_role': 'Lieutenant Governor',
            'previous_location': 'San Francisco',
        },
        'type': EpisodeType.json,
        'description': 'political profile',
    },
    {
        'content': {
            'name': 'Gavin Newsom',
            'position': 'Governor',
            'term_start': 'January 7, 2019',
            'term_end': 'Present',
        },
        'type': EpisodeType.json,
        'description': 'term information',
    },
]

# Neo4j connection class to manage lifecycle
class Neo4jConnectionManager:
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.graphiti = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Graphiti connection with Neo4j"""
        if self._initialized:
            return self.graphiti
        
        try:
            # Create Graphiti with Neo4j connection
            self.graphiti = Graphiti(
                self.uri, 
                self.user, 
                self.password,
                llm_client=GeminiClient(
                    config=LLMConfig(
                        api_key=gemini_api_key,
                        model="gemini-2.0-flash"
                    )
                ),
                embedder=GeminiEmbedder(
                    config=GeminiEmbedderConfig(
                        api_key=gemini_api_key,
                        embedding_model="embedding-001"
                    )
                )
            )
            
            # Initialize indices
            await self.graphiti.build_indices_and_constraints()
            self._initialized = True
            return self.graphiti
        except Exception as e:
            # Get full traceback for debugging
            error_tb = traceback.format_exc()
            print(f"Neo4j connection error: {error_tb}")
            raise Exception(f"Failed to initialize Neo4j connection: {str(e)}")
    
    async def close(self):
        """Close the Neo4j connection"""
        if self.graphiti:
            try:
                await self.graphiti.close()
                self._initialized = False
                self.graphiti = None
            except Exception as e:
                print(f"Error closing Neo4j connection: {str(e)}")
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self.graphiti
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()


# Create connection manager
@st.cache_resource(show_spinner=False)
def get_connection_manager():
    """Get or create the Neo4j connection manager"""
    return Neo4jConnectionManager(neo4j_uri, neo4j_user, neo4j_password)

async def add_episodes(graphiti, episodes):
    """Add episodes to the knowledge graph"""
    try:
        for i, episode in enumerate(episodes):
            # Convert JSON content to string if needed
            content = episode['content']
            if not isinstance(content, str) and episode['type'] == EpisodeType.json:
                content = json.dumps(content)
            
            # Add episode with proper error handling
            await graphiti.add_episode(
                name=f'Episode {i}',
                episode_body=content,
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
        return True
    except Exception as e:
        # Log full traceback for debugging
        error_tb = traceback.format_exc()
        print(f"Error adding episodes: {error_tb}")
        raise Exception(f"Error adding episodes: {str(e)}")

async def search_graph(graphiti, query, center_node_uuid=None):
    """Search the knowledge graph for relationships"""
    try:
        return await graphiti.search(query, center_node_uuid=center_node_uuid)
    except Exception as e:
        error_tb = traceback.format_exc()
        print(f"Error searching graph: {error_tb}")
        raise Exception(f"Error searching graph: {str(e)}")

async def search_nodes(graphiti, query, limit=5):
    """Search for nodes in the knowledge graph"""
    try:
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = limit
        results = await graphiti._search(query=query, config=node_search_config)
        return results.nodes
    except Exception as e:
        error_tb = traceback.format_exc()
        print(f"Error searching nodes: {error_tb}")
        raise Exception(f"Error searching nodes: {str(e)}")

def create_visualization_data(results):
    """
    Create visualization data from search results that properly represents 
    the Graphiti knowledge graph structure with NO UUIDs visible
    
    Args:
        results: List of search results from Graphiti
        
    Returns:
        Dictionary containing nodes and edges for visualization
    """
    nodes = []
    edges = []
    node_map = {}  # Map UUID to node objects with labels
    
    for result in results:
        # Extract source node information
        source_uuid = result.source_node_uuid
        
        # Process source node
        if source_uuid not in node_map:
            # Extract meaningful node name/label
            source_label = "Entity"
            source_name = "Unnamed Entity"  # Default fallback name, never show UUID
            
            # Check for name/label attributes (many possible attribute names)
            potential_name_attrs = [
                'source_node_name', 'source_name', 'source_node_summary', 'source_summary',
                'entity_name', 'name', 'title'
            ]
            
            # Try all potential name attributes
            for attr in potential_name_attrs:
                if hasattr(result, attr) and getattr(result, attr):
                    name_value = getattr(result, attr)
                    if isinstance(name_value, str) and name_value:
                        source_name = name_value[:25] + "..." if len(name_value) > 25 else name_value
                        break
            
            # Extract entity type/label
            potential_type_attrs = [
                'source_node_labels', 'source_entity_type', 'source_type', 'entity_type', 'type'
            ]
            
            # Try all potential type attributes
            for attr in potential_type_attrs:
                if hasattr(result, attr) and getattr(result, attr):
                    type_value = getattr(result, attr)
                    if isinstance(type_value, list) and len(type_value) > 0:
                        source_label = type_value[0]
                        break
                    elif type_value:
                        source_label = str(type_value)
                        break
            
            # Absolutely ensure no UUIDs in display
            if source_uuid in source_name:
                source_name = f"{source_label} Entity"
                
            # Store node in map
            node_map[source_uuid] = {
                'id': source_uuid,  # Internal ID only, never displayed
                'label': f"{source_name}",  # Just entity name, no UUID
                'full_label': f"{source_label}: {source_name}",  # Full info for tooltip
                'title': f"{source_label}: {source_name}\n\n{result.fact if hasattr(result, 'fact') else ''}",
                'color': '#4CAF50',  # Green for source nodes
                'shape': 'ellipse',
                'entity_type': source_label
            }
        
        # Extract target node information
        target_uuid = result.target_node_uuid
        
        # Process target node
        if target_uuid not in node_map:
            # Extract meaningful node name/label
            target_label = "Entity"
            target_name = "Unnamed Entity"  # Default fallback name, never show UUID
            
            # Check for name/label attributes (many possible attribute names)
            potential_name_attrs = [
                'target_node_name', 'target_name', 'target_node_summary', 'target_summary',
                'entity_name', 'name', 'title'
            ]
            
            # Try all potential name attributes
            for attr in potential_name_attrs:
                if hasattr(result, attr) and getattr(result, attr):
                    name_value = getattr(result, attr)
                    if isinstance(name_value, str) and name_value:
                        target_name = name_value[:25] + "..." if len(name_value) > 25 else name_value
                        break
            
            # Extract entity type/label
            potential_type_attrs = [
                'target_node_labels', 'target_entity_type', 'target_type', 'entity_type', 'type'
            ]
            
            # Try all potential type attributes
            for attr in potential_type_attrs:
                if hasattr(result, attr) and getattr(result, attr):
                    type_value = getattr(result, attr)
                    if isinstance(type_value, list) and len(type_value) > 0:
                        target_label = type_value[0]
                        break
                    elif type_value:
                        target_label = str(type_value)
                        break
            
            # Absolutely ensure no UUIDs in display
            if target_uuid in target_name:
                target_name = f"{target_label} Entity"
                
            # Store node in map
            node_map[target_uuid] = {
                'id': target_uuid,  # Internal ID only, never displayed
                'label': f"{target_name}",  # Just entity name, no UUID
                'full_label': f"{target_label}: {target_name}",  # Full info for tooltip
                'title': f"{target_label}: {target_name}\n\n{result.fact if hasattr(result, 'fact') else ''}",
                'color': '#2196F3',  # Blue for target nodes
                'shape': 'ellipse',
                'entity_type': target_label
            }
        
        # Extract relationship type
        relationship_type = "related to"
        
        # Check for relationship type attributes (try multiple possible attribute names)
        potential_rel_attrs = [
            'relationship_type', 'edge_type', 'relation_type', 'rel_type', 'predicate'
        ]
        
        # Try all potential relationship attributes
        for attr in potential_rel_attrs:
            if hasattr(result, attr) and getattr(result, attr):
                rel_value = getattr(result, attr)
                if isinstance(rel_value, str) and rel_value:
                    relationship_type = rel_value
                    break
        
        # If no relationship type found, try to extract from fact text if available
        if relationship_type == "related to" and hasattr(result, 'fact') and result.fact:
            # Try to extract relationship verb (common pattern: "Entity1 RELATION Entity2")
            source_name_clean = node_map[source_uuid]['label'].lower()
            target_name_clean = node_map[target_uuid]['label'].lower()
            fact = result.fact.lower()
            
            # Remove source and target from fact to isolate relationship
            if source_name_clean in fact and target_name_clean in fact:
                # Find positions of source and target in the fact
                try:
                    source_pos = fact.index(source_name_clean)
                    target_pos = fact.index(target_name_clean)
                    
                    # Extract the text between them (if source comes before target)
                    if source_pos < target_pos:
                        between_text = fact[source_pos + len(source_name_clean):target_pos].strip()
                        if between_text and len(between_text.split()) <= 3:
                            relationship_type = between_text
                except ValueError:
                    pass  # If we can't find the names in the fact, just use the default
        
        # Create edge with descriptive relationship
        edges.append({
            'id': f"{source_uuid}_{target_uuid}",
            'from': source_uuid,
            'to': target_uuid,
            'label': relationship_type,  # Use relationship type as visible label
            'title': result.fact if hasattr(result, 'fact') else f"Relationship: {relationship_type}",
            'arrows': 'to',
            'font': {'align': 'middle'},
            'length': 200  # Slightly longer connections for better readability
        })
    
    # Convert node map to list for visualization
    nodes = list(node_map.values())
    
    return {"nodes": nodes, "edges": edges}

def visualize_graph(nodes, edges):
    """
    Create a network visualization of the knowledge graph with improved styling
    
    Args:
        nodes: List of node objects for visualization
        edges: List of edge objects for visualization
        
    Returns:
        HTML string of the visualization
    """
    net = Network(height="500px", width="100%", directed=True, notebook=True)
    
    # Set physics and interaction options for better visualization
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 25
        }
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "continuous"
        },
        "color": {
          "inherit": true
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "font": {
          "size": 10,
          "face": "arial",
          "align": "middle"
        }
      },
      "nodes": {
        "font": {
          "size": 13,
          "face": "arial"
        },
        "borderWidth": 2,
        "scaling": {
          "min": 10,
          "max": 30
        },
        "shape": "dot"
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "hover": true
      }
    }
    """)
    
    # Add nodes
    for node in nodes:
        net.add_node(
            node['id'], 
            label=node['label'], 
            title=node.get('title', ''),
            color=node.get('color', '#97c2fc')
        )
    
    # Add edges
    for edge in edges:
        net.add_edge(
            edge['from'], 
            edge['to'], 
            label=edge.get('label', ''),
            title=edge.get('title', ''),
            arrows=edge.get('arrows', 'to')
        )
    
    # Save and return the HTML
    html_file = "knowledge_graph.html"
    net.save_graph(html_file)
    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()
    
    return html

# ===================== UI COMPONENTS =====================
def render_search_results(results):
    """Render search results in the UI"""
    if results:
        st.subheader("Search Results")
        for i, result in enumerate(results):
            with st.expander(f"Result {i+1}: {result.fact[:50]}..."):
                st.write(f"**Fact:** {result.fact}")
                st.write(f"**UUID:** {result.uuid}")
                if hasattr(result, 'valid_at') and result.valid_at:
                    st.write(f"**Valid from:** {result.valid_at}")
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    st.write(f"**Valid until:** {result.invalid_at}")
        return True
    return False

def render_node_results(nodes):
    """Render node search results in the UI"""
    if nodes:
        st.subheader("Node Search Results")
        for i, node in enumerate(nodes):
            with st.expander(f"Node {i+1}: {node.name}"):
                st.write(f"**Name:** {node.name}")
                st.write(f"**UUID:** {node.uuid}")
                st.write(f"**Labels:** {', '.join(node.labels)}")
                st.write(f"**Summary:** {node.summary}")
                st.write(f"**Created At:** {node.created_at}")
                if hasattr(node, 'attributes') and node.attributes:
                    st.write("**Attributes:**")
                    for key, value in node.attributes.items():
                        st.write(f"- {key}: {value}")
        return True
    return False

# ===================== MAIN APP FUNCTION =====================
async def main():
    """Main async function for Streamlit app"""
    
    # Title and description
    st.title("Graphiti Knowledge Graph Explorer")
    st.markdown("Build, query, and visualize knowledge graphs with Neo4j and Google Gemini")
    
    # Create two columns: 2/3 for main content, 1/3 for sidebar
    main_col, sidebar_col = st.columns([2, 1])
    
    # Get the connection manager
    connection_manager = get_connection_manager()
    
    # ===================== RIGHT SIDEBAR =====================
    with sidebar_col:
        # Graph Visualization (top right)
        st.subheader("Knowledge Graph Visualization")
        
        if st.session_state.visualization_data["nodes"]:
            html = visualize_graph(
                st.session_state.visualization_data["nodes"], 
                st.session_state.visualization_data["edges"]
            )
            st.components.v1.html(html, height=400)
        else:
            st.info("Run a search to generate visualization")
        
        # Custom Episode Input
        st.subheader("Add Custom Episode")
        custom_episode_type = st.selectbox("Type", ["Text", "JSON"], key="custom_type")
        custom_episode_description = st.text_input("Description", "custom data", key="custom_desc")
        
        if custom_episode_type == "Text":
            custom_episode_content = st.text_area("Content", "", key="custom_content")
        else:
            custom_episode_content = st.text_area("Content (JSON)", "{}", key="custom_content_json")
        
        add_custom = st.button("Add Custom Episode")
        if add_custom:
            if custom_episode_content:
                custom_episode = {
                    'content': custom_episode_content,
                    'type': EpisodeType.text if custom_episode_type == "Text" else EpisodeType.json,
                    'description': custom_episode_description
                }
                
                # If JSON, validate and parse
                if custom_episode_type == "JSON":
                    try:
                        custom_episode['content'] = json.loads(custom_episode_content)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON. Please check your input.")
                        st.stop()
                
                sample_episodes.append(custom_episode)
                st.success("Custom episode added")
            else:
                st.warning("Please enter content")
    
    # ===================== MAIN CONTENT =====================
    with main_col:
        # Initialize Knowledge Graph
        st.header("1. Initialize Knowledge Graph")
        init_col1, init_col2 = st.columns([1, 1])
        
        with init_col1:
            init_button = st.button("Initialize Database", disabled=st.session_state.graph_initialized)
        
        with init_col2:
            if st.session_state.graph_initialized:
                st.success("Graph database initialized")
            elif st.session_state.status_message:
                st.info(st.session_state.status_message)
        
        if init_button and not st.session_state.graph_initialized:
            with st.status("Initializing graph database...", expanded=True) as status:
                st.session_state.status_message = "Connecting to Neo4j..."
                try:
                    # Initialize Neo4j connection asynchronously
                    graphiti = await connection_manager.initialize()
                    
                    # Update session state
                    st.session_state.graph_initialized = True
                    st.session_state.status_message = "Database initialized successfully"
                    status.update(label="✅ Database Initialized", state="complete")
                except Exception as e:
                    st.session_state.status_message = f"Error: {str(e)}"
                    status.update(label="❌ Initialization Failed", state="error")
                    st.error(str(e))
        
        # Add Episodes Section
        st.header("2. Add Episodes to Graph")
        if not st.session_state.graph_initialized:
            st.warning("Please initialize the graph database first")
        else:
            episodes_col1, episodes_col2 = st.columns([1, 1])
            
            with episodes_col1:
                with st.expander("View Sample Episodes"):
                    for i, episode in enumerate(sample_episodes):
                        st.markdown(f"**Episode {i+1}**")
                        if episode['type'] == EpisodeType.text:
                            st.text(episode['content'])
                        else:
                            st.json(episode['content'])
                        st.caption(f"Type: {episode['type'].value}, Description: {episode['description']}")
                        st.markdown("---")
            
            with episodes_col2:
                add_episodes_button = st.button("Add Episodes to Graph", disabled=st.session_state.episodes_added)
                if st.session_state.episodes_added:
                    st.success("Episodes added to graph")
            
            if add_episodes_button and not st.session_state.episodes_added:
                with st.status("Adding episodes to graph...", expanded=True) as status:
                    try:
                        # Use the connection manager to get Graphiti instance
                        async with connection_manager as graphiti:
                            # Add episodes asynchronously
                            success = await add_episodes(graphiti, sample_episodes)
                            
                            # Update session state
                            if success:
                                st.session_state.episodes_added = True
                                status.update(label="✅ Episodes Added Successfully", state="complete")
                            else:
                                status.update(label="❌ Failed to Add Episodes", state="error")
                    except Exception as e:
                        status.update(label="❌ Error Adding Episodes", state="error")
                        st.error(str(e))
        
        # Search Graph Section
        st.header("3. Search Knowledge Graph")
        if not st.session_state.episodes_added:
            st.warning("Please add episodes to the graph first")
        else:
            search_type = st.radio("Search Type", ["Relationship Search", "Node Search"], horizontal=True)
            search_query = st.text_input("Search Query", "Who was the California Attorney General?")
            search_button = st.button("Search Graph")
            
            if search_button:
                # Clear previous results
                st.session_state.search_results = []
                st.session_state.node_search_results = []
                st.session_state.reranked_results = []
                
                with st.status(f"Searching for: {search_query}", expanded=True) as status:
                    try:
                        # Use the connection manager to get Graphiti instance
                        async with connection_manager as graphiti:
                            if search_type == "Relationship Search":
                                # Perform relationship search
                                results = await search_graph(graphiti, search_query)
                                st.session_state.search_results = results
                                
                                # Create visualization data
                                if results:
                                    st.session_state.visualization_data = create_visualization_data(results)
                                
                                status.update(label="✅ Search Complete", state="complete")
                            else:  # Node Search
                                # Perform node search
                                nodes = await search_nodes(graphiti, search_query)
                                st.session_state.node_search_results = nodes
                                status.update(label="✅ Node Search Complete", state="complete")
                    except Exception as e:
                        status.update(label="❌ Search Failed", state="error")
                        st.error(str(e))
            
            # Display search results if available
            if st.session_state.search_results:
                # Render search results
                has_results = render_search_results(st.session_state.search_results)
                
                if has_results:
                    # Option to rerank results
                    st.subheader("Rerank Results")
                    results = st.session_state.search_results
                    
                    center_node_index = st.selectbox(
                        "Select a result to use as center node", 
                        range(len(results)),
                        format_func=lambda i: f"Result {i+1}: {results[i].fact[:50]}..."
                    )
                    
                    rerank_button = st.button("Rerank Results")
                    
                    if rerank_button:
                        with st.status("Reranking results...", expanded=True) as status:
                            try:
                                # Use the connection manager to get Graphiti instance
                                async with connection_manager as graphiti:
                                    # Get center node UUID
                                    center_node_uuid = results[center_node_index].source_node_uuid
                                    
                                    # Perform reranking
                                    reranked_results = await search_graph(graphiti, search_query, center_node_uuid)
                                    st.session_state.reranked_results = reranked_results
                                    
                                    # Update visualization if results available
                                    if reranked_results:
                                        st.session_state.visualization_data = create_visualization_data(reranked_results)
                                    
                                    status.update(label="✅ Reranking Complete", state="complete")
                            except Exception as e:
                                status.update(label="❌ Reranking Failed", state="error")
                                st.error(str(e))
            
            # Display reranked results if available
            if st.session_state.reranked_results:
                render_search_results(st.session_state.reranked_results)
            
            # Display node search results if available
            if st.session_state.node_search_results:
                render_node_results(st.session_state.node_search_results)
    
    # Footer
    st.markdown("---")
    st.caption("Powered by Graphiti, Neo4j, and Google Gemini")

# Run the async app
if __name__ == "__main__":
    asyncio.run(main())
else:
    # Special handling for Streamlit's execution model
    # Create a new event loop and use nest_asyncio for nested loops
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # This allows using asyncio operations inside a streamlit app
    nest_asyncio.apply()

    # Handle Neo4j connections with clean cleanup
    try:
        result = asyncio.run(main())
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        # Try to close any open connections
        connection_manager = get_connection_manager()
        if hasattr(connection_manager, 'graphiti') and connection_manager.graphiti:
            try:
                asyncio.run(connection_manager.close())
            except:
                pass