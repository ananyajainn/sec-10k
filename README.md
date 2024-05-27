# sec-10k
download_10k_files.py:
- an attempt that downloads the SEC 10-k filings based on user input, puts them into a vector database via Pinecone,
  and uses OpenAI to analyze the data and return text insights
gemini-10kfilings.py:
- uses gemini on a single 10-k text file and produces text insights to console 
llama-10kfilings.py:
- uses LLaMA via HuggingFace to analyze the downloaded files
