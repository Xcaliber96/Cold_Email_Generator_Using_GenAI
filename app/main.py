import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import logging

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://www.accenture.com/in-en/careers/jobdetails?id=ATCI-4154481-S1608969_en")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            st.info("Loading data from URL...")
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)

            st.info("Loading portfolio...")
            portfolio.load_portfolio()

            st.info("Extracting jobs...")
            jobs = llm.extract_jobs(data)
            logger.info(f"Extracted jobs: {jobs}")

            # Check if there are any jobs extracted
            if jobs:
                job = jobs[0]  # Focus only on the first job
                skills = job.get('skills', [])
                
                # You can remove or comment out the following lines to prevent showing skills
                # st.subheader("Selected Job")
                # st.write("Skills:", skills)

                try:
                    # Remove the line that outputs skills
                    # st.info(f"Querying links for skills: {skills}")

                    # Directly query the links without printing them
                    links = portfolio.query_links(skills)

                    # Remove the line that outputs links
                    # st.write("Links:", links)
                except ValueError as e:
                    st.error(f"Error querying links: {e}")
                    logger.error(f"Error querying links: {e}", exc_info=True)
                    return  # Exit early if there's an error

                st.info("Generating email...")
                email = llm.write_mail(job, links)
                
                # Output only the email body
                st.subheader("Here is the email body:")
                st.code(email, language='markdown')

            else:
                st.warning("No jobs found in the provided URL.")

        except Exception as e:
            st.error(f"An Error Occurred: {e}")
            logger.error(f"An error occurred: {e}", exc_info=True)
            st.exception(e)

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
