# app.py

import streamlit as st
import anywidget
import traitlets
from st_supabase_connection import SupabaseConnection, execute_query
from streamlit_anywidget import anywidget as render_widget
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────
# 1 · ChatWidget Definition
# ────────────────────────────────────────────────────
class ChatWidget(anywidget.AnyWidget):
    _esm = """
    // Load Supabase's ESM module dynamically (version-pinned) and export an async render
    export default {
      async render({ model, el }) {
        // Dynamically import the ESM build with named exports
        const { createClient } = await import(
          'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm'
        );
        const url      = model.get('url');
        const key      = model.get('key');
        const username = model.get('username');
        const supabase = createClient(url, key);

        console.log('ChatWidget initialized');

        // Container for messages
        const msgs = document.createElement('div');
        msgs.style.height = '300px';
        msgs.style.overflowY = 'auto';
        msgs.style.border = '1px solid #ccc';
        msgs.style.padding = '8px';
        msgs.style.marginBottom = '8px';
        el.appendChild(msgs);

        // Subscribe to new messages
        supabase
          .from('messages')
          .on('INSERT', payload => {
            console.log('New message received:', payload.new);
            const div = document.createElement('div');
            div.textContent = `${payload.new.username}: ${payload.new.text}`;
            msgs.appendChild(div);
            msgs.scrollTop = msgs.scrollHeight;
          })
          .subscribe();

        // Build input form
        const form = document.createElement('form');
        form.style.display = 'flex';
        form.style.gap = '4px';

        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = 'Type a message…';
        input.style.flex = '1';

        const btn = document.createElement('button');
        btn.textContent = 'Send';
        btn.type = 'submit';

        form.append(input, btn);
        el.appendChild(form);

        form.addEventListener('submit', async e => {
          e.preventDefault();
          const text = input.value.trim();
          if (text) {
            console.log('Sending message:', text);
            await supabase.from('messages').insert([{ username, text }]);
            input.value = '';
          }
        });
      }
    };
    """

    url      = traitlets.Unicode().tag(sync=True)
    key      = traitlets.Unicode().tag(sync=True)
    username = traitlets.Unicode().tag(sync=True)

# ────────────────────────────────────────────────────
# 1 · Chat Display Fragment
# ────────────────────────────────────────────────────
@st.fragment(run_every=2)
def chat_fragment(st_supabase_client):
    """
    This fragment fetches the latest 50 messages from Supabase
    every 2 seconds and displays them in the main area.
    """
    # Build and execute the query
    history = execute_query(
        st_supabase_client.table("messages")
            .select("username, text")
            .order("created_at", desc=True)
            .limit(50),
    )

    if history and history.data:
        for row in history.data:
            st.write(f"**{row['username']}**: {row['text']}")

# ────────────────────────────────────────────────────
# 2 · Streamlit App
# ────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Real-Time Supabase Chat", layout="centered")
    logger.info("Starting Streamlit app")

    # 2.1 · Google OAuth guard
    if not st.experimental_user.is_logged_in:
        st.header("🔐 Log in with Google")
        if st.button("Log in"):
            logger.info("User attempting to log in")
            st.login()
        return

    logger.info(f"User logged in: {st.experimental_user.name}")

    # 2.2 · Sidebar: two message inputs
    st.sidebar.header("Send a message as…")
    st_supabase_client = st.connection(
        name="supabase",
        type=SupabaseConnection,
        ttl=None,
        url="https://zlmwlnozwjjxzikukidm.supabase.co",
        key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpsbXdsbm96d2pqeHppa3VraWRtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU1MjkxNzIsImV4cCI6MjA2MTEwNTE3Mn0.Au1BVBfJuHZMvxky2vWtrYMXU_PmGAuuOrvnwIlrg6k"
    )

    # 2.3 · Sidebar: Send as User A or User B
    st.sidebar.header("Send a message")
    msg_a = st.sidebar.text_input("User A", key="a")
    if st.sidebar.button("Send as User A"):
        st_supabase_client.table("messages") \
                .insert([{"username": "User A", "text": msg_a}]) \
                .execute()  # insert via supabase-py :contentReference[oaicite:5]{index=5}

    st.sidebar.markdown("---")
    msg_b = st.sidebar.text_input("User B", key="b")
    if st.sidebar.button("Send as User B"):
        st_supabase_client.table("messages") \
                .insert([{"username": "User B", "text": msg_b}]) \
                .execute()  # same insert pattern :contentReference[oaicite:7]{index=7}

    # 2.4 · Render the chat fragment in main area
    with st.sidebar:
        st.title("💬 Chatroom")
        with st.container(height=300, border=True):
            chat_fragment(st_supabase_client)  # this fragment reruns every 2s and on message send :contentReference[oaicite:9]{index=9}

    # Logout button
    st.sidebar.markdown("---")
    st.sidebar.write(f"👋 Logged in as: {st.experimental_user.name}")
    if st.sidebar.button("Log out"):
        logger.info("User logging out")
        st.logout()
        st.rerun()

    # 2.3 · Main area: title + history + real-time widget
    st.title("🚀 Real-Time Supabase Chatroom")

    # 2.5 · Instantiate & configure the widget
    logger.info("Initializing ChatWidget")
    chat = ChatWidget()
    chat.url      = "https://zlmwlnozwjjxzikukidm.supabase.co"
    chat.key      = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpsbXdsbm96d2pqeHppa3VraWRtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU1MjkxNzIsImV4cCI6MjA2MTEwNTE3Mn0.Au1BVBfJuHZMvxky2vWtrYMXU_PmGAuuOrvnwIlrg6k"
    chat.username = st.experimental_user.name

    # 2.6 · Render the AnyWidget
    logger.info("Rendering ChatWidget")
    render_widget(chat)


if __name__ == "__main__":
    main()
