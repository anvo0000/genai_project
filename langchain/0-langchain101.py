#Follow the instructions at https://python.langchain.com/docs/integrations/chat/openai/
# Requirements:
# langchain-openai==0.3.0
# langchain==0.3.24

import dotenv
import os

dotenv.load_dotenv()

# If you install langchain version v0.2 might be got some error related to verbose, debug, llm_cache
# Uncomment below lines of code to resolve it

# import langchain
# langchain.verbose = False
# langchain.debug = False
# langchain.llm_cache = False

from langchain_openai import ChatOpenAI
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,      # how creative/random you want responses
    api_key=os.getenv("OPENAI_API_KEY")
)
response = chat.invoke("Tell me about the hidden langchain folder in MacOS")
print(response.content)

#### Response.content: ####
# In macOS, when you use LangChain or similar Python libraries, certain configurations, caches, or data may be stored in hidden folders. These hidden folders typically start with a dot (`.`) and are not visible by default in Finder.
#
# For LangChain specifically, if you have installed it and used it to create or manage language models, you might find a hidden folder related to it in your home directory. This folder could be used for storing cached models, configurations, or other runtime data.
#
# To view hidden folders in Finder, you can use the following steps:
#
# 1. Open Finder.
# 2. Press `Command + Shift + .` (the period key). This will toggle the visibility of hidden files and folders.
#
# Once you can see hidden files, you can navigate to your home directory (usually `/Users/your_username/`) and look for any folders related to LangChain, which might be named something like `.langchain` or similar.
#
# If you want to access this folder via the terminal, you can do so by using the `cd` command. For example:
#
# ```bash
# cd ~/.langchain
# ```
#
# This command will take you to the LangChain folder if it exists.
#
# Keep in mind that the exact name and contents of the folder may vary based on the version of LangChain you are using and how you've configured it. Always be cautious when modifying or deleting files in hidden folders, as they may be essential for the proper functioning of the library.
