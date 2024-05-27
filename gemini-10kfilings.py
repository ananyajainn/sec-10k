from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain_experimental.llms import ChatLlamaAPI
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key='AIzaSyDZGZYtv2hKPoUz_4K3qo__Z-gdMI0uccY')

# Map
map_template = """The following is a set of documents
{docs}
Based on this document, provide a detailed analysis of the company's financial performance, risk factors, and business strategy. 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)


reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated, detailed analysis of the company's financial performance, risk factors, and business strategy. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

reduce_prompt = hub.pull("rlm/map-prompt")

# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=100000,
)


# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

loader = TextLoader(
    './file 1.txt',
)
docs = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=10000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

print(map_reduce_chain.invoke(split_docs)['output_text'])