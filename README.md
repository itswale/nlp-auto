# NLP UI Automation

A Streamlit-based UI testing tool that utilizes Playwright and spaCy for natural language processing-driven automation (NLP). Deployed on [Render](https://render.com/), this app allows users to build and run test flows for web applications.

## Features

- **Simplified Mode**: Pre-built templates for common tasks (e.g., search, form filling).
- **Advanced Mode**: Custom Playwright actions (e.g., click, fill, visit).
- **Screenshots**: Capture results of successful tests.
- **Analytics**: View test execution statistics.

## Prerequisites

- Python 3.12
- Docker (for local or custom deployment)
- Git

## Local Setup

Follow these steps to run the app on your machine after cloning or forking the repository:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/nlp-ui-automation.git
   cd nlp-ui-automation
2. **Install Dependencies: Using Docker (recommended, includes system dependencies)**:
      ```bash
   docker build -t nlp-auto .
   docker run -p 8501:8501 nlp-auto
   ```
   **Without Docker (requires manual system dependency installation)**:
   Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Install Playwright browsers:
   ```bash
   python -m playwright install chromium
   ```
   Install system dependencies (Ubuntu/Debian example):
   ```bash
   sudo apt-get update && sudo apt-get install -y \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 \
    libasound2 libatspi2.0-0
   ```
   Run the app:
   ```bash
   streamlit run app.py
   ```

3. **Access the App**:
Open http://localhost:8501 in your browser.

## Deployment on Render
The app is deployed on Render using a Dockerfile for system dependencies.
1. Fork or Clone: Fork this repository or clone it to your GitHub account.
2. Set Up Render:
      Sign up at Render.
      Create a new Web Service and connect your GitHub repository.
      Set the runtime to Docker.
      No additional settings are needed (the Dockerfile handles everything).
3. Deploy: Access the provided URL (e.g., https://nlp-auto.onrender.com).
   
   **Environment Variables (optional)**
      ```bash
      Add CLOUD_ENV=true in Render’s dashboard under "Environment" for cloud-specific behavior.
      ```
## Usage
   1. Build Test Flow: Use Simplified or Advanced mode to create test suites.
   Run Tests: Click "Run All Test Suites" to execute and view results/analytics.
   2. Headless Mode: Enable in the sidebar (required for cloud).
   
## Files
   app.py: Main application code.
   requirements.txt: Python dependencies.
   Dockerfile: Docker setup for Render and local runs.
   ui_test_report.log: Logs test execution (auto-generated).
   screenshots/: Stores test screenshots (auto-generated).

## Notes
   Playwright requires system libraries (e.g., libnss3). The Dockerfile ensures these are present on Render or locally.
   spaCy’s en_core_web_sm model is downloaded on the first run if not present.
   Render’s free tier (750 hours/month) is sufficient for personal use.

## Contributing
   Feel free to fork the repository, make changes, and submit a pull request. Issues and feature requests are welcome!

## License
   This project is licensed under the MIT License. See the LICENSE file for details.
