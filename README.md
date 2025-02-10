# 🌱 ParselyFi - Your All-in-One Financial Data & AI Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parselyfi.streamlit.app)

**ParselyFi** is a Streamlit application designed to streamline financial research and workflows for venture capital, middle market, and public company analysis. It integrates a suite of tools including a file manager for S3-compatible storage, an AI assistant for workflow automation, and a public financial data dashboard, all in one place.

## ✨ Key Features

*   **S3 File Manager:**
    *   Browse, upload, download, create folders, and delete files directly from your S3-compatible storage (optimized for Supabase Storage).
    *   Pagination for efficient handling of large file lists.
    *   Intuitive folder navigation and file selection.
*   **Parsely AI Assistant:**
    *   Chat-based AI assistant for help with government forms, workflow automation, and general inquiries.
    *   Clear chat history and system message display.
*   **Public Dashboard:**
    *   Explore curated financial data from various sources (Company Data, Youtube Transcriptions, News, Forums).
    *   Interactive data tables with filtering and different views (Master DB, Products, Partnerships, Investors).
    *   Data fetched from Supabase database with caching for performance.
*   **File Works Tab:**
    *   Preview selected files directly within the app (PDF, CSV, Excel, Text-based formats).
    *   Supports various file types for viewing and analysis.
*   **Company Search & Analysis:** (Placeholder - Feature in Development)
    *   Intended for future integration of company-specific search and analysis tools.
*   **News & YouTube Tab:** (Placeholder - Feature in Development)
    *   Planned to display financial news alerts and daily YouTube transcription reports.
*   **Transcription & Summaries Tab:** (Placeholder - Feature in Development)
    *   Future feature for transcribing uploaded audio files and generating summaries.

## 🛠️ Technologies Used

*   **Streamlit:** For building the interactive web application UI.
*   **Supabase:**
    *   For backend database (PostgreSQL) to store financial data and user data.
    *   Supabase Storage (S3-compatible) for file management.
    *   Supabase Python Client for database and storage interactions.
*   **boto3:** Python SDK for interacting with AWS S3 (used for Supabase Storage).
*   **pandas:** For data manipulation and display in dataframes.
*   **os, math, base64, io, datetime:** Standard Python libraries for various functionalities.
*   **Potentially: OpenAI's GPT-3 (or similar LLMs):** For future AI assistant enhancements.

## 🚀 Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_GITHUB_REPOSITORY_URL]
    cd ParselyFi
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(It's recommended to use a virtual environment like `venv` or `conda`)*
3.  **Set up Supabase:**
    *   Create a Supabase project at [https://supabase.com/](https://supabase.com/).
    *   Create a Supabase Storage bucket.
    *   Create the necessary database tables (SQL code provided in the application code comments - `supabase_sql_code_for_tables.sql`).
4.  **Configure Secrets:**
    *   Create a `.streamlit/secrets.toml` file in your project directory.
    *   Add your Supabase credentials and S3 storage details to `secrets.toml` as follows (replace with your actual values):

        ```toml
        [supabase]
        SUPABASE_URL = "YOUR_SUPABASE_URL"
        SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY" # Use anon key for frontend access, consider service key for backend
        SUPABASE_S3_BUCKET_NAME = "YOUR_SUPABASE_STORAGE_BUCKET_NAME"
        SUPABASE_S3_ENDPOINT_URL = "YOUR_SUPABASE_STORAGE_ENDPOINT_URL"
        SUPABASE_S3_BUCKET_REGION = "YOUR_SUPABASE_STORAGE_REGION"
        SUPABASE_S3_BUCKET_ACCESS_KEY = "YOUR_SUPABASE_STORAGE_ACCESS_KEY"
        SUPABASE_S3_BUCKET_SECRET_KEY = "YOUR_SUPABASE_STORAGE_SECRET_KEY"
        ```
5.  **Run the Streamlit application:**
    ```bash
    streamlit run your_streamlit_app_filename.py # Replace with your main script filename (e.g., app.py)
    ```
6.  **Login:** The application uses Streamlit's experimental user login. Log in with your Google account when prompted.

## 🧑‍💻 Usage Instructions

1.  **Sidebar Navigation:** Use the sidebar to navigate between features:
    *   **File Manager:** Browse your S3 bucket, manage files and folders.
    *   **AI Assistant:** Interact with the Parsely AI chatbot.
    *   **System Messages:** View system notifications and logs.
2.  **Public Dashboard Tab:** Explore the public financial data dashboards. Use the outer and inner tabs to navigate through different datasets and views.
3.  **File Works Tab:** Select files in the File Manager sidebar. Go to the "File Works" tab in the main area to preview and interact with selected files.
4.  **Action Buttons:** Utilize the buttons in the File Manager to create folders, upload files, and delete items.
5.  **Pagination:** For file lists and data tables, use the pagination controls to navigate through pages of data.

## 🤝 Contributing (Optional)

[If you are open to contributions, add information here about how others can contribute. For example:]

Contributions are welcome! Please fork the repository and submit pull requests with your improvements. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License (Optional)

[If you have a license, specify it here. For example:]

[MIT License](LICENSE)

## 👤 About the Creator

**Homen Shum** - Data-Driven Professional | AI/ML, Data Analytics, Workflow Automation

*   [Personal Website](https://homenshum.com/)
*   [LinkedIn Profile](https://linkedin.com/in/homen-shum)

---
*Last Updated: Feb 2025*