import requests
from bs4 import BeautifulSoup
import ollama
import json
from langchain_ollama import ChatOllama
import re

MODEL = "llama3.2"

class llms:
    def __init__(self, model, company, website, url):
        self.model = model
        self.website = website
        self.company = company
        self.url = url
    
    self.llm = ChatOllama(
        model = self.model,
        temperature = 0.8,
        num_predict = 256,
        # other params ...
        )
    
    def get_relevant_links(self):
        system_prompt = """You are provided with a list of links found on a webpage.\n \
        You should only include links that are relevant to the company brochure,\n \
        such as links to an About page, or a Company page, or products, or solution pages.\n
        You should respond in JSON as the example below. I don't need you give any explanation, just a JSON response.\n:
            "links": [
                {"type": "about page", "url": "https://full.url/goes/here/about"},
                {"type": "careers page": "url": "https://another.full.url/careers"}
            ]
        """
    
        user_prompt = f"""Here is the company website of {self.url}.
            I will provide you with a list of links found on the website as below: {self.website.get_links(self.url)} \n
            Please decide which of these are relevant web links for a company brochure, respond with the full https URL in JSON format.
            Step by step review the solutions or products and include them the relevant links.
            Do not include Terms of Service, Privacy, email links."""

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        response = llm.invoke(messages=payload["messages"], stream=payload["stream"])
        print(response)
        result = response["message"]["content"]
        return result["links"]

    def extract_urls(text):
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(pattern, text)
        return urls

    def get_all_details(self):
        relevant_links = self.get_relevant_links()
        urls = extract_urls(relevant_links)

        details = []
        for link in urls:
            content = self.website.get_content(link["url"])
            details.append({"type": link["type"], "content": content})
        return details

    def get_brochure_user_prompt(company_name):
        user_prompt = f"""You are looking at a company called: {self.company_name}\n
        Use the below information to build a short brochure of the company in markdown.\n
        The company's website is: {self.website.url}\n
        The wesite's contents are: {self.get_all_details()}"""
        return user_prompt


class Website:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
            # Add more headers as needed
        }

    def get_raw_html(self, url):
        response = requests.get(url, headers=self.headers)
        body = response.content
        soup = BeautifulSoup(body, 'html.parser')
        return soup

    def get_title(self, url):
        soup = self.get_raw_html(url)
        return soup.title.string if soup.title else "No title found"

    def get_content(self, url):
        soup = self.get_raw_html(url)
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            text = soup.body.get_text(separator="\n", strip=True)
        else:
            text = ""
        return text

    def get_links(self, url):
        soup = self.get_raw_html(url)
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
        return f"\n".join(self.links) + "\n\n"

    
    
url = "https://adurolife.com/"
website_crape = Website()
# print(website_crape.get_title(url))
# print(website_crape.get_content(url))
# print(website_crape.get_links(url))

llms_instance = llms(model=MODEL, company='Aduro',website=website_crape, url=url)
print(llms_instance.get_relevant_links())
# print(llms_instance.get_all_details())