# v10.4.2 - Enhanced selector detection using descriptions
# Changes:
# 1. Improved detect_selector to map descriptions to selectors dynamically
# 2. Fixed HH suite Step 3 failure with better Facebook email detection
# 3. Kept all Playwright actions and robust error handling

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
from dataclasses import dataclass
from typing import List, Dict, Optional
import hashlib

# Load NLP model
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please run `python -m spacy download en_core_web_sm` to install the NLP model.")
    st.stop()

# Set up logging
logging.basicConfig(filename='ui_test_report.log', level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.screenshots_dir = "screenshots"
        self.screenshot_buffer = {}
        self.loop = None
        os.makedirs(self.screenshots_dir, exist_ok=True)

    async def detect_selector(self, page, action, description, value=None):
        desc_lower = description.lower()
        logger.debug(f"Detecting selector - Action: {action}, Desc: {description}, Value: {value}")
        
        # Parse description with NLP
        doc = nlp(desc_lower)
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"]]

        # Site-specific optimizations
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

        # General dynamic selector detection
        frames = [page] + page.frames[1:]
        for frame in frames:
            try:
                # Input fields (Fill, Enter)
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

                # Buttons (Click)
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

                # Checkboxes
                elif "checkbox" in keywords or action.lower() in ["check", "uncheck"]:
                    return "input[type='checkbox']", frame

                # Dropdowns
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

        # For 'visit' action, if URL is missing, try to use the initial URL from session_state
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
                elif action == "hover":
                    await frame.wait_for_selector(selector, state="visible", timeout=5000)
                    await frame.hover(selector)
                    result["logs"].append(f"Hovered over {selector}")
                elif action == "check":
                    await frame.wait_for_selector(selector, state="visible", timeout=5000)
                    await frame.check(selector)
                    result["logs"].append(f"Checked {selector}")
                elif action == "uncheck":
                    await frame.wait_for_selector(selector, state="visible", timeout=5000)
                    await frame.uncheck(selector)
                    result["logs"].append(f"Unchecked {selector}")
                elif action == "select":
                    if not value:
                        raise ValueError("Value required")
                    await frame.wait_for_selector(selector, state="visible", timeout=5000)
                    await frame.select_option(selector, value=value)
                    result["logs"].append(f"Selected '{value}' in {selector}")
                elif action == "wait":
                    wait_time = int(value) if value and value.isdigit() else 1000
                    await page.wait_for_timeout(wait_time)
                    result["logs"].append(f"Waited for {wait_time}ms")
                elif action == "see":
                    if not value:
                        raise ValueError("Value required")
                    is_visible = await page.is_visible(f"text={value}")
                    result["logs"].append(f"Checked visibility of '{value}': {is_visible}")
                    if not is_visible:
                        raise Exception(f"Text '{value}' not visible")

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
                await page.screenshot(path=screenshot_path, full_page=True)
                result["screenshot"] = screenshot_path
                with open(screenshot_path, "rb") as f:
                    screenshot_data = f.read()
                    self.screenshot_buffer[f"{suite_name}_{description}"] = screenshot_data
                    result["screenshot_hash"] = hashlib.md5(screenshot_data).hexdigest()
            except Exception as e:
                logger.error(f"Screenshot failed: {str(e)}")
                result["logs"].append("Screenshot failed")

        result["execution_time"] = (datetime.now() - start_time).total_seconds()
        self.results.append(result)
        logger.info(f"Step {step_idx + 1} - Completed {action} - Status: {result['status']}, Logs: {result['logs']}")
        return result["status"] in ["passed", "skipped"], None, None, None

    async def run_suite(self, suite: TestSuite, enable_screenshots=True, start_index=0, retries=2):
        p = await async_playwright().start()
        browser = await p.chromium.launch(headless=st.session_state.get("headless", False))
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        st.session_state[f"browser_{suite.name}"] = browser
        st.session_state[f"context_{suite.name}"] = context
        st.session_state[f"page_{suite.name}"] = page
        self.loop = asyncio.get_running_loop()

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

    def style_dataframe(self, df):
        def color_status(val):
            color = 'green' if val == 'passed' else 'red' if val == 'failed' else 'gray'
            return f'background-color: {color}'
        return df.style.map(color_status, subset=['status'])

    def render(self):
        st.title("NLP UI Automation")

        with st.sidebar:
            st.header("Guide & Controls")
            st.markdown("""
                ### Quick Guide
                - **Simplified Mode**: Quick templates for common tasks.
                - **Advanced Mode**: Full control with all Playwright actions.
            """)
            mode = st.radio("Mode", ["Simplified", "Advanced"], index=1, help="Choose mode for test creation.")
            enable_screenshots = st.checkbox("Enable Screenshots", value=True, help="Capture screenshots for passed steps.")
            headless = st.checkbox("Run Headless", value=False, help="Run without visible browser window.")
            retries = st.number_input("Failure Retries", min_value=0, max_value=5, value=2, help="Retries for failed actions.")
            st.session_state["headless"] = headless
            if st.button("Clear Test Flow", help="Reset all suites and results"):
                if "running_suites" in st.session_state and st.session_state.running_suites:
                    asyncio.run(self.executor._cleanup_all_browsers(st.session_state.get("test_suites", [])))
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

            suite_name = st.text_input("Suite Name", key="suite_name", help="Unique name for your test suite.")
            url = st.text_input("Initial Website URL", "https://www.fb.com", key="url_input", help="Starting URL for the suite.")

            if mode == "Simplified":
                with st.form(key="simple_form"):
                    template = st.selectbox("Select Template", ["Search and Submit", "Click Link", "Fill Form"], help="Predefined action templates.")
                    description = st.text_input("Description", help="Describe the action (e.g., 'Email').")
                    if template == "Search and Submit":
                        search_term = st.text_input("Search Term", help="Text to search for.")
                        submit = st.form_submit_button("Add Template")
                        if submit and suite_name and description and search_term:
                            suite = TestSuite(suite_name, url, [TestCase("Fill", description, search_term, press_enter=True)])
                            st.session_state.test_suites.append(suite)
                            st.success(f"Added {template}")
                            st.rerun()
                    elif template == "Click Link":
                        link_text = st.text_input("Link Text", help="Text of the link to click.")
                        submit = st.form_submit_button("Add Template")
                        if submit and suite_name and description and link_text:
                            suite = TestSuite(suite_name, url, [TestCase("Click", description, link_text)])
                            st.session_state.test_suites.append(suite)
                            st.success(f"Added {template}")
                            st.rerun()
                    elif template == "Fill Form":
                        field_value = st.text_input("Field Value", help="Value to enter in the field.")
                        submit = st.form_submit_button("Add Template")
                        if submit and suite_name and description and field_value:
                            suite = TestSuite(suite_name, url, [TestCase("Fill", description, field_value, press_enter=True)])
                            st.session_state.test_suites.append(suite)
                            st.success(f"Added {template}")
                            st.rerun()

            elif mode == "Advanced":
                with st.form(key="advanced_form"):
                    action = st.selectbox("Action", ["Visit", "Click", "Fill", "Enter", "Hover", "Check", "Uncheck", "Select", "Wait", "See"], 
                                        help="Choose an action: Visit URL, Click element, Fill field, etc.")
                    description = st.text_input("Description", help="Describe the action (e.g., 'Email' for email field).")
                    value = st.text_input("Value", help="Value for Fill, Enter, Select, See, or Wait time (ms).")
                    press_enter = st.checkbox("Press Enter", value=False, help="Press Enter after Fill/Enter action.")
                    manual_selector = st.text_input("Manual Selector (optional)", help="CSS selector if automatic detection fails.")
                    depends_on = st.number_input("Depends On Step (0 for none)", min_value=0, value=0, help="Step this depends on.")
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
                                st.success(f"Added {action}")
                            else:
                                suite = TestSuite(suite_name, url, [new_case])
                                st.session_state.test_suites.append(suite)
                                st.success(f"Created suite with {action}")
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
                                    if st.button("X", key=f"delete_{suite_idx}_{step_idx}", help="Remove this step"):
                                        if step_idx == 0 and len(suite.test_cases) > 0:
                                            suite.test_cases.pop(0)
                                        elif step_idx > 0:
                                            suite.test_cases.pop(step_idx - 1)
                                        st.rerun()

                if st.button("Run All Test Suites", help="Execute all test suites"):
                    with st.spinner("Running tests..."):
                        completed = asyncio.run(self.executor.run_tests(st.session_state.test_suites, enable_screenshots=enable_screenshots, retries=retries))
                    if completed:
                        st.success("All tests completed! Swtich tab to view results")
                        if self.executor.results:
                            df = pd.DataFrame(self.executor.results)
                            total_steps = len(df)
                            pass_count = len(df[df["status"] == "passed"])
                            fail_count = len(df[df["status"] == "failed"])
                            skipped_count = len(df[df["status"] == "skipped"])
                            st.write(f"Total Steps: {total_steps}")
                            st.write(f"Passed: {pass_count}")
                            st.write(f"Failed: {fail_count}")
                            st.write(f"Skipped: {skipped_count}")
                    else:
                        st.warning("Some tests failed or were interrupted. Switch tab to view result")

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
                st.info("No tests run yet. Please run tests to see detailed results.")

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
                st.info("No tests run yet. Please run tests to see analytics.")

def main():
    executor = TestExecutor()
    ui = UIManager(executor)
    ui.render()

if __name__ == "__main__":
    main()
