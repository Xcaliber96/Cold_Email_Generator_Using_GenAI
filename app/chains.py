import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()  # Correct path to your .env file

class Chain:
    def __init__(self):
        # Get the GROQ_API_KEY from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Make sure it is set in the .env file.")
        
        # Use the environment variable instead of hardcoding the key
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,  # API key fetched from environment variable
            model_name='llama3-8b-8192'
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            The `skills` field must be a non-empty list of strings.
            If you can't extract specific skills, include general skills related to the job role.
            Return only the valid JSON array of job postings. Do not include any explanations or additional text.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        
        try:
            # Try to parse the entire response as JSON
            parsed_res = json.loads(res.content)
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the text
            try:
                json_start = res.content.index('[')
                json_end = res.content.rindex(']') + 1
                json_str = res.content[json_start:json_end]
                parsed_res = json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                # If JSON extraction fails, return a default job
                parsed_res = [{
                    "role": "Software Developer",
                    "experience": "Not specified",
                    "skills": ["Software Development", "Problem Solving", "Communication"],
                    "description": "Unable to parse job details"
                }]
        
        # Ensure the result is a list
        if not isinstance(parsed_res, list):
            parsed_res = [parsed_res]
        
        # Ensure each job has a non-empty skills list
        for job in parsed_res:
            if 'skills' not in job or not job['skills']:
                job['skills'] = ["Software Development", "Problem Solving", "Communication"]
        
        return parsed_res

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Aaditya, a Software developer engineer at PQR. PQR is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of PQR 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase PQR portfolio: {link_list}
            Remember you are Aaditya, SDE at PQR. 
            Write the email body only. Do not include any salutations, signatures, or additional explanations.
            ### EMAIL BODY:
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content.strip()


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
