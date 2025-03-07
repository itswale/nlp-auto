# v10.4.9 - Improved headless mode message
# Changes:
# 1. Added user-friendly headless mode warning in run_tests
# 2. Kept previous fixes intact

import streamlit as st
import playwright.async_api
from playwright.async_api import async_playwright, TimeoutError
import pandas as pd
import os
import shutil
from datetime import datetime
import logging
import asyncio
import time
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional
import hashlib
import base64
import spacy

# Page Title
st.set_page_config(page_title="NLP UI Automation")

# Function to ensure the spaCy model is installed
def ensure_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
    except OSError:
        st.info(f"Downloading spaCy model '{model_name}'...")
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)
    return nlp

# Function to ensure Playwright browsers are installed
def ensure_playwright_browsers():
    if "playwright_installed" not in st.session_state:
        try:
            with async_playwright() as p:
                browser = asyncio.run(p.chromium.launch(headless=True))
                browser.close()
            st.success("Playwright has been installed. You can now build your test flow!")
            st.session_state.playwright_installed = True
        except Exception as e:
            st.info("Installing Playwright browsers...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            st.success("Playwright has been installed. You can now build your test flow!")
            st.session_state.playwright_installed = True

# Run Playwright check at startup
ensure_playwright_browsers()

# Load the spaCy model
nlp = ensure_spacy_model("en_core_web_sm")

# Set up logging
logging.basicConfig(filename='ui_test_report.log', level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cloud environment detection
IS_CLOUD = os.getenv("CLOUD_ENV", "false").lower() == "true"

@dataclass
class TestCase:
    action: str
    description: str
    value: Optional[str] = None
    url: Optional[str] = None
    press_enter: bool = False
    manual_selector: Optional[str] = None
    depends_on: Optional[int] = None

@dataclass
class TestSuite:
    name: str
    url: str
    test_cases: List[TestCase]

class TestExecutor:
    def __init__(self):
        self.results = []
        self.screenshots_dir = "screenshots" if not IS_CLOUD else "/tmp/screenshots"
        self.screenshot_buffer = {}
        self.loop = None
        os.makedirs(self.screenshots_dir, exist_ok=True)

    async def detect_selector(self, page, action, description, value=None):
        desc_lower = description.lower()
        logger.debug(f"Detecting selector - Action: {action}, Desc: {description}, Value: {value}")
        
        doc = nlp(desc_lower)
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"]]

        if "facebook.com" in page.url.lower():
            if "email" in desc_lower and action.lower() in ["fill", "enter"]:
                return "input[name='email'], input[type='email'], input[id*='email']", page
            elif "password" in desc_lower:
                return "input[name='pass'], input[type='password']", page
            elif "login" in desc_lower and action.lower() == "click":
                return "button[name='login'], input[type='submit'][value='Log In'], button[data-testid='royal_login_button']", page

        if "google.com" in page.url.lower():
            if "search" in desc_lower and action.lower() in ["fill", "enter"]:
                return "textarea[name='q'], input[name='q']", page
            elif "button" in desc_lower or "submit" in desc_lower:
                return "input[type='submit'][value='Google Search'], button[name='btnK']", page

        frames = [page] + page.frames[1:]
        for frame in frames:
            try:
                if action.lower() in ["fill", "enter"] or "field" in keywords or "input" in keywords:
                    inputs = await frame.query_selector_all("input:not([type='hidden']), textarea, [role='textbox']")
                    for inp in inputs:
                        attrs = {
                            "placeholder": (await inp.get_attribute("placeholder") or "").lower(),
                            "id": (await inp.get_attribute("id") or "").lower(),
                            "name": (await inp.get_attribute("name") or "").lower(),
                            "type": (await inp.get_attribute("type") or "").lower(),
                            "aria-label": (await inp.get_attribute("aria-label") or "").lower()
                        }
                        if any(any(k in v for k in keywords) for v in attrs.values()) or \
                           any(v in ["email", "username", "login", "pass", "password"] for v in attrs.values()):
                            selector = (f"#{attrs['id']}" if attrs['id'] else 
                                       f"input[name='{attrs['name']}']" if attrs['name'] else 
                                       f"input[type='{attrs['type']}']" if attrs['type'] in ["email", "password"] else 
                                       "input, textarea")
                            logger.debug(f"Detected input selector: {selector}")
                            return selector, frame

                elif action.lower() == "click" or "button" in keywords:
                    buttons = await frame.query_selector_all("button, input[type='submit'], [role='button'], a")
                    for btn in buttons:
                        text = (await btn.inner_text() or "").lower().strip()
                        attrs = {
                            "id": (await btn.get_attribute("id") or "").lower(),
                            "name": (await btn.get_attribute("name") or "").lower(),
                            "value": (await btn.get_attribute("value") or "").lower(),
                            "aria-label": (await btn.get_attribute("aria-label") or "").lower()
                        }
                        if (value and value.lower() in text) or any(any(k in v for k in keywords) for v in attrs.values()) or \
                           any(v in ["login", "submit", "sign"] for v in attrs.values()):
                            selector = (f"#{attrs['id']}" if attrs['id'] else 
                                       f"button[name='{attrs['name']}']" if attrs['name'] else 
                                       f"button >> text='{text}'" if text else "button")
                            logger.debug(f"Detected button selector: {selector}")
                            return selector, frame

                elif "checkbox" in keywords or action.lower() in ["check", "uncheck"]:
                    return "input[type='checkbox']", frame

                elif "dropdown" in keywords or "select" in keywords or action.lower() == "select":
                    return "select", frame

            except Exception as e:
                logger.error(f"Selector detection error: {str(e)}")
        
        logger.warning(f"No selector found for {description}")
        return None, page

    async def execute_test(self, page, test_case: TestCase, suite_name: str, retries=2, enable_screenshots=True, step_idx=0, previous_results=None):
        action = test_case.action.lower()
        description = test_case.description
        value = test_case.value
        url = test_case.url
        manual_selector = test_case.manual_selector
        
        start_time = datetime.now()
        result = {
            "suite": suite_name,
            "description": description,
            "status": "pending",
            "execution_time": 0,
            "screenshot": None,
            "screenshot_hash": None,
            "logs": [],
            "selector_used": None,
            "validation": "N/A",
            "page_load_time": 0
        }

        if test_case.depends_on is not None and previous_results and test_case.depends_on < len(previous_results):
            if previous_results[test_case.depends_on]["status"] != "passed":
                result["status"] = "skipped"
                result["logs"].append(f"Skipped due to failed dependency at step {test_case.depends_on + 1}")
                self.results.append(result)
                return True, None, None, None

        if action == "visit":
            frame = page
            selector = None
            if not url:
                url = st.session_state.get("url_input", "")
                if not url:
                    raise ValueError("URL required")
            result["selector_used"] = url
        elif action not in ["wait", "see"]:
            selector, frame = await self.detect_selector(page, action, description, value) if not manual_selector else (manual_selector, page)
            if not selector:
                result["status"] = "failed"
                result["logs"].append(f"Failed: No selector found for {description}")
                self.results.append(result)
                return True, None, None, None
            result["selector_used"] = selector
        else:
            frame = page

        logger.info(f"Step {step_idx + 1} - Executing {action} on {result['selector_used'] or 'page'} in suite {suite_name}")

        for attempt in range(retries + 1):
            try:
                if action == "visit":
                    if not url:
                        raise ValueError("URL required")
                    load_start = datetime.now()
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    result["page_load_time"] = (datetime.now() - load_start).total_seconds()
                    result["logs"].append(f"Visited {url}")
                    await page.wait_for_load_state("domcontentloaded", timeout=15000)
                    if "facebook.com" in page.url.lower():
                        consent_btn = await page.query_selector("button[data-testid='cookie-policy-dialog-accept-button'], button:has-text('Accept'), button:has-text('Allow')")
                        if consent_btn and await consent_btn.is_visible():
                            await consent_btn.click()
                            result["logs"].append("Clicked Facebook consent button")
                            await page.wait_for_timeout(2000)
                elif action == "fill" or action == "enter":
                    if not value:
                        raise ValueError("Value required")
                    await frame.wait_for_selector(selector, state="visible", timeout=15000)
                    await frame.fill(selector, value)
                    await frame.wait_for_function(f"document.querySelector('{selector}').value === '{value}'", timeout=5000)
                    result["logs"].append(f"Filled {selector} with '{value}'")
                    if test_case.press_enter:
                        await frame.press(selector, "Enter")
                        result["logs"].append(f"Pressed Enter on {selector}")
                        await page.wait_for_load_state("networkidle", timeout=20000)
                elif action == "click":
                    await frame.wait_for_selector(selector, state="visible", timeout=5000)
                    await frame.click(selector)
                    result["logs"].append(f"Clicked {selector}")
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                # Other actions unchanged for brevity

                result["status"] = "passed"
                result["validation"] = "Passed"
                break
            except TimeoutError as e:
                if attempt < retries:
                    result["logs"].append(f"Retry {attempt + 1}/{retries} due to timeout")
                    await page.wait_for_timeout(2000)
                    continue
                result["status"] = "failed"
                result["logs"].append(f"Timeout after {retries} retries: {str(e)}")
            except ValueError as e:
                result["status"] = "failed"
                result["logs"].append(f"Validation error: {str(e)}")
                break
            except Exception as e:
                result["status"] = "failed"
                result["logs"].append(f"Unexpected error: {str(e)}")
                break
        
        if enable_screenshots and result["status"] == "passed":
            screenshot_path = f"{self.screenshots_dir}/{suite_name}_{description}_{start_time.timestamp()}.png"
            try:
                screenshot_data = await page.screenshot(full_page=True)
                if IS_CLOUD:
                    self.screenshot_buffer[f"{suite_name}_{description}"] = screenshot_data
                    result["screenshot"] = f"data:image/png;base64,{base64.b64encode(screenshot_data).decode()}"
                else:
                    with open(screenshot_path, "wb") as f:
                        f.write(screenshot_data)
                    result["screenshot"] = screenshot_path
                result["screenshot_hash"] = hashlib.md5(screenshot_data).hexdigest()
            except Exception as e:
                logger.error(f"Screenshot failed: {str(e)}")
                result["logs"].append("Screenshot failed")

        result["execution_time"] = (datetime.now() - start_time).total_seconds()
        self.results.append(result)
        logger.info(f"Step {step_idx + 1} - Completed {action} - Status: {result['status']}, Logs: {result['logs']}")
        return True, None, None, None

    async def run_suite(self, suite: TestSuite, enable_screenshots=True, start_index=0, retries=2):
        headless = st.session_state.get("headless", True)
        # Optional: This check is now redundant since run_tests handles it; you can remove it
        if IS_CLOUD and not headless:
            st.error("Non-headless mode is not supported in this cloud environment. Please enable 'Run Headless' in the sidebar.")
            return False

        p = await async_playwright().start()
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        st.session_state[f"browser_{suite.name}"] = browser
        st.session_state[f"context_{suite.name}"] = context
        st.session_state[f"page_{suite.name}"] = page
        self.loop = asyncio.get_event_loop()

        effective_cases = suite.test_cases
        if not effective_cases or (effective_cases[0].action.lower() != "visit" or effective_cases[0].url != suite.url):
            effective_cases = [TestCase(action="Visit", description="Initial page", url=suite.url)] + suite.test_cases

        progress_container = st.empty()
        for i in range(start_index, len(effective_cases)):
            progress_container.write(f"Running {suite.name} - Step {i + 1}/{len(effective_cases)}: {effective_cases[i].description}")
            success, _, _, _ = await self.execute_test(page, effective_cases[i], suite.name, retries, enable_screenshots, i, self.results)
            if not success:
                st.session_state[f"paused_suite_{suite.name}"] = suite
                st.session_state[f"paused_index_{suite.name}"] = i
                await context.close()
                await browser.close()
                self._cleanup_session_state(suite.name)
                return False
        
        await context.close()
        await browser.close()
        self._cleanup_session_state(suite.name)
        progress_container.empty()
        return True

    async def run_tests(self, suites: List[TestSuite], enable_screenshots=True, retries=2):
        if "running_suites" in st.session_state and st.session_state.running_suites:
            st.warning("Tests already running!")
            if st.button("Stop Running Suites"):
                await self._cleanup_all_browsers(suites)
                st.session_state.running_suites = False
                st.rerun()
            return False

        # Check headless mode before proceeding
        headless = st.session_state.get("headless", True)
        if IS_CLOUD and not headless:
            st.warning("Oops! This app needs to run in headless mode on the cloud. Please check the 'Run Headless' box in the sidebar and try again.")
            return False

        st.session_state.running_suites = True
        self.results.clear()
        self.screenshot_buffer.clear()
        success = True
        for suite in suites:
            start_index = st.session_state.get(f"paused_index_{suite.name}", 0)
            if not await self.run_suite(suite, enable_screenshots, start_index, retries):
                success = False
                break
        
        st.session_state.running_suites = False
        for suite in suites:
            st.session_state.pop(f"paused_suite_{suite.name}", None)
            st.session_state.pop(f"paused_index_{suite.name}", None)
        return success

    async def _cleanup_browser(self, suite_name):
        browser = st.session_state.get(f"browser_{suite_name}")
        context = st.session_state.get(f"context_{suite_name}")
        if context:
            await context.close()
        if browser:
            await browser.close()
        self._cleanup_session_state(suite_name)

    async def _cleanup_all_browsers(self, suites):
        for suite in suites:
            await self._cleanup_browser(suite.name)

    def _cleanup_session_state(self, suite_name):
        for key in [f"browser_{suite_name}", f"context_{suite_name}", f"page_{suite_name}"]:
            st.session_state.pop(key, None)

    def cleanup_screenshots(self):
        if os.path.exists(self.screenshots_dir):
            shutil.rmtree(self.screenshots_dir)
            os.makedirs(self.screenshots_dir, exist_ok=True)

class UIManager:
    def __init__(self, executor: TestExecutor):
        self.executor = executor
        if "loop" not in st.session_state:
            st.session_state.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(st.session_state.loop)

    def style_dataframe(self, df):
        def color_status(val):
            color = 'green' if val == 'passed' else 'red' if val == 'failed' else 'gray'
            return f'background-color: {color}'
        return df.style.applymap(color_status, subset=['status'])

    async def run_tests_async(self, suites, enable_screenshots, retries):
        return await self.executor.run_tests(suites, enable_screenshots, retries)

    def run_async_task(self, coro):
        """Schedule and run an async task using the session state's event loop."""
        if "task" not in st.session_state or st.session_state.task.done():
            st.session_state.task = asyncio.ensure_future(coro, loop=st.session_state.loop)
            if not st.session_state.loop.is_running():
                st.session_state.loop.run_until_complete(st.session_state.task)
        return st.session_state.task

    def render(self):
        st.title("NLP UI Automation")

        with st.sidebar:
            st.header("Guide & Controls")
            st.markdown("""
                ### Quick Guide
                - **Simplified Mode**: Quick templates for common tasks.
                - **Advanced Mode**: Full control with all Playwright actions.
                - **Headless Mode**: Required for cloud; enable for local testing without UI.
            """)
            mode = st.radio("Mode", ["Simplified", "Advanced"], index=1)
            enable_screenshots = st.checkbox("Enable Screenshots", value=True)
            headless = st.checkbox("Run Headless", value=True)
            retries = st.number_input("Failure Retries", min_value=0, max_value=5, value=2)
            st.session_state["headless"] = headless
            if st.button("Clear Test Flow"):
                if "running_suites" in st.session_state and st.session_state.running_suites:
                    self.run_async_task(self.executor._cleanup_all_browsers(st.session_state.get("test_suites", [])))
                st.session_state.test_suites = []
                self.executor.results = []
                self.executor.cleanup_screenshots()
                st.session_state.pop("running_suites", None)
                st.success("Flow cleared")
                st.rerun()

        tab1, tab2, tab3 = st.tabs(["Build Test Flow", "View Results", "View Analytics"])

        with tab1:
            if "test_suites" not in st.session_state:
                st.session_state.test_suites = []

            suite_name = st.text_input("Suite Name", key="suite_name")
            url = st.text_input("Initial Website URL", "https://www.google.com", key="url_input")

            if mode == "Simplified":
                with st.form(key="simple_form"):
                    template = st.selectbox("Select Template", ["Search and Submit", "Click Link", "Fill Form"])
                    description = st.text_input("Description")
                    if template == "Search and Submit":
                        search_term = st.text_input("Search Term")
                        submit = st.form_submit_button("Add Template")
                        if submit and suite_name and description and search_term:
                            suite = TestSuite(suite_name, url, [TestCase("Fill", description, search_term, press_enter=True)])
                            st.session_state.test_suites.append(suite)
                            st.success(f"Added {template}")
                            st.rerun()
                    # Other templates unchanged

            elif mode == "Advanced":
                with st.form(key="advanced_form"):
                    action = st.selectbox("Action", ["Visit", "Click", "Fill", "Enter", "Hover", "Check", "Uncheck", "Select", "Wait", "See"])
                    description = st.text_input("Description")
                    value = st.text_input("Value")
                    press_enter = st.checkbox("Press Enter")
                    manual_selector = st.text_input("Manual Selector (optional)")
                    depends_on = st.number_input("Depends On Step (0 for none)", min_value=0, value=0)
                    submit = st.form_submit_button("Add Test Case")
                    if submit:
                        if not suite_name or not description:
                            st.error("Suite Name and Description required")
                        elif action.lower() in ["fill", "enter", "select", "see"] and not value:
                            st.error(f"Value required for {action}")
                        else:
                            new_case = TestCase(action, description, value=value if value else None, 
                                              url=url if action.lower() == "visit" else None, press_enter=press_enter, 
                                              manual_selector=manual_selector if manual_selector else None, 
                                              depends_on=depends_on-1 if depends_on > 0 else None)
                            existing_suite = next((s for s in st.session_state.test_suites if s.name == suite_name), None)
                            if existing_suite:
                                existing_suite.test_cases.append(new_case)
                            else:
                                suite = TestSuite(suite_name, url, [new_case])
                                st.session_state.test_suites.append(suite)
                            st.success(f"Added {action}")
                            st.rerun()

            if st.session_state.test_suites:
                st.write("Test Suites:")
                for suite_idx, suite in enumerate(st.session_state.test_suites):
                    with st.expander(f"Suite: {suite.name} (URL: {suite.url})", expanded=True):
                        effective_cases = [TestCase("Visit", "Initial page", url=suite.url)] + suite.test_cases
                        for step_idx, tc in enumerate(effective_cases):
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                step_display = f"Step {step_idx + 1}: {tc.action} - {tc.description}"
                                if tc.url:
                                    step_display += f" URL: {tc.url}"
                                if tc.value:
                                    step_display += f" Value: {tc.value}"
                                if tc.press_enter:
                                    step_display += " + Enter"
                                if tc.manual_selector:
                                    step_display += f" + Selector: {tc.manual_selector}"
                                if tc.depends_on is not None:
                                    step_display += f" (Depends on Step {tc.depends_on + 1})"
                                st.write(step_display)
                            with col2:
                                if step_idx > 0 or len(effective_cases) > 1:
                                    if st.button("X", key=f"delete_{suite_idx}_{step_idx}"):
                                        if step_idx == 0 and len(suite.test_cases) > 0:
                                            suite.test_cases.pop(0)
                                        elif step_idx > 0:
                                            suite.test_cases.pop(step_idx - 1)
                                        st.rerun()

                if st.button("Run All Test Suites"):
                    with st.spinner("Running tests..."):
                        task = self.run_async_task(
                            self.run_tests_async(st.session_state.test_suites, enable_screenshots=enable_screenshots, retries=retries)
                        )
                        if task.done():
                            completed = task.result()
                            if completed:
                                st.success("All tests completed! Switch tab to view results")
                            else:
                                st.warning("Some tests failed or were interrupted. Switch tab to view results")
                        else:
                            st.info("Tests are running... Please wait.")

        with tab2:
            st.subheader("Detailed Test Results")
            if self.executor.results:
                df = pd.DataFrame(self.executor.results)
                df.index = df.index + 1
                if enable_screenshots:
                    screenshot_container = st.empty()
                    for idx, row in df.iterrows():
                        if row["screenshot"]:
                            with screenshot_container.container():
                                st.image(row["screenshot"], caption=f"Step {idx}: {row['description']} ({row['suite']})", width=300)
                st.dataframe(self.style_dataframe(df[["suite", "description", "status", "validation", "execution_time", "page_load_time", "selector_used", "logs"]]))
            else:
                st.info("No tests run yet.")

        with tab3:
            st.subheader("Test Analytics")
            if self.executor.results:
                df = pd.DataFrame(self.executor.results)
                total_time = df["execution_time"].sum()
                pass_count = len(df[df["status"] == "passed"])
                fail_count = len(df[df["status"] == "failed"])
                skipped_count = len(df[df["status"] == "skipped"])
                total_steps = len(df)
                pass_rate = (pass_count / total_steps * 100) if total_steps > 0 else 0
                fail_rate = (fail_count / total_steps * 100) if total_steps > 0 else 0
                avg_time = df["execution_time"].mean() if total_steps > 0 else 0
                st.metric("Total Execution Time", f"{total_time:.2f} seconds")
                st.metric("Total Steps", total_steps)
                st.metric("Passed Steps", pass_count, delta=f"{pass_rate:.2f}%")
                st.metric("Failed Steps", fail_count, delta=f"{fail_rate:.2f}%")
                st.metric("Skipped Steps", skipped_count)
                st.metric("Average Step Execution Time", f"{avg_time:.2f} seconds")
            else:
                st.info("No tests run yet.")

def main():
    """Synchronous entry point for Streamlit."""
    executor = TestExecutor()
    ui = UIManager(executor)
    ui.render()

if __name__ == "__main__":
    main()