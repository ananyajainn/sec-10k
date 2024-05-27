from openai import OpenAI

from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import pinecone 
from pinecone import Pinecone, ServerlessSpec


client = OpenAI(api_key='sk-proj-yBBMLMkTlrjBY91vvDOJT3BlbkFJkkPupV1VsQchV2OFG6Xd')

#pinecone = Pinecone(api_key="e7a5b73e-76ef-4a30-a5c6-f3572b7d655c")

# Initialize Pinecone
pinecone = Pinecone(api_key='e7a5b73e-76ef-4a30-a5c6-f3572b7d655c')
index_name = 'sec-10k-analysis7'

# Delete existing index (if needed)
if index_name in pinecone.list_indexes():
    print("deletng")
    pinecone.delete_index(index_name)

# Define the index spec
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
        )
    )

# Create the index if it does not exist

index = pinecone.Index(index_name)

# takes in, ticket, start and end year and downalods 10k filings
def download_10k_filings(ticker, start_year, end_year):
    d1 = Downloader("Placeholder", "myemail@gmail.com")
    d1.get("10-K", ticker, after=f"{start_year}-01-01", before=f"{end_year}-12-31")

# extracts text from file using BeautifulSoup to parse
def extract_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        text = soup.get_text()
    return text

# get embeddings
def get_embeddings(text):
    response = client.embeddings.create(input = [text],
    model="text-embedding-ada-002")
    return response.data[0].embedding

# store text
def store_text(text, metadata):
    embeddings = get_embeddings(text)
    index.upsert(vectors=[{
        'id': metadata['id'],
        'values': embeddings,
        'metadata': metadata
    }])

# analyze using GPT3
def analyze(text):
    response = client.completions.create(
        model = "gpt-3.5-turbo",
        prompt=f"Analyze the following SEC 10-K filing text and provide insights on the company's financial performance, risk factors, and business strategy:\n\n{text}\n\nInsights:",
        max_tokens=100
    )
    return response.choices[0].text.strip()

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
        for file in files:
            if file.endswith('.html'):
                file_path = os.path.join(root, file)
                text = extract_text(file_path)
                metadata = {
                    'id': os.path.basename(file_path),
                    'ticker': ticker,
                    'year': start_year,  # Adjust how to extract the year if necessary
                }
                store_text(text, metadata)
                print(f"Stored text from {file_path} to Pinecone.")

    #Example query to analyze stored texts
    query_text = "Analyze recent financial performance and risk factors."
    query_embeddings = get_embeddings(query_text)

    result = index.query(queries=[query_embeddings], top_k=1)
    if result and 'matches' in result and result['matches']:
        best_match = result['matches'][0]
        file_path = best_match['metadata']['id']
        text = extract_text(os.path.join(filings_path, file_path))
        insights = analyze(text)
        print(f"\nInsights from {file_path}:")
        print(insights)



if __name__ == "__main__":
    main()