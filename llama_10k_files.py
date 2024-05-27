import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from llamaapi import LlamaAPI

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

llama = LlamaAPI('LL-HAuZiEhcPbnkSiepD51H1HOfunmNQZAvPbE5MtSP7t3DpeBzY8vhk7aTuD3BWRnw')

# takes in, ticket, start and end year and downalods 10k filings
def download_10k_filings(ticker, start_year, end_year):
    d1 = Downloader("Placeholder", "myemail@gmail.com")
    d1.get("10-K", ticker, after=f"{start_year}-01-01", before=f"{end_year}-12-31")

# extracts text from file using BeautifulSoup to pars
def extract_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        text = soup.get_text()
    return text


# Function to read files from the directory and perform analysis
def analyze(file_path):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# main
def main():
    # get info
    # eliminates extra spaces and puts all in uppercase to format
    ticker = input("Enter company ticker: ").strip().upper()
    # type cast to int and elimante extra spaces
    start_year = int(input("Enter start year(YYYY): ").strip())
    end_year = int(input("Enter the end year(YYYY): ").strip())

    # check if years are valid

    # download filings
    print(f"Downloading 10-K filings for {ticker} from {start_year} to {end_year}...")
    download_10k_filings(ticker, start_year, end_year)
    print("Download complete.")

    filings_dir = 'sec_edgar_filings'

     # Construct full path to the directory containing 10-K filings
    filings_path = os.path.join(os.getcwd(), 'sec-edgar-filings')

    # Verify the directory and its contents
    print(f"Checking directory: {filings_path}")
    if not os.path.exists(filings_path):
        print(f"Directory '{filings_path}' does not exist or is empty.")
        return

    for root, dirs, files in os.walk(filings_path):
        print("made it")
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_path}")
                text = extract_text(file_path)
                summary = analyze(text)
                print(summary)  # Print or process the summary as needed



if __name__ == "__main__":
    main()