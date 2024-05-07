import cmd
import os
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
## Theres currently a warning that comes up when importing the parser, just ignore it. https://github.com/langflow-ai/langflow/issues/1820
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()



def init_llm_fallback():
    # Preamble
    preamble = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""

    # LLM
    llm_fallback = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # Prompt
    prompt = lambda x: ChatPromptTemplate.from_messages(
        [
            ("system", preamble),
            ("human", "{question}")
        ]
    )

    # Chain
    global llm_chain_fallback
    llm_chain_fallback = prompt | llm_fallback | StrOutputParser()

    ## Run
    #question = "Hi how are you?"
    #generation = llm_chain.invoke({"question": question})
    #if should_print_logs: print(generation)
    return llm_chain_fallback

def init_llm_fallback_with_context():
    # Preamble
    preamble = """You are a capable first principles thinker with vast knowledge of the world. Answer the question based primarily on the provided context, but you should also use your knowledge and sharp reasoning."""

    # LLM
    llm_fallback_with_context = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # Prompt
    prompt = lambda x: ChatPromptTemplate.from_messages(
        [
            ("system", preamble),
            ("human",  "context: \n\n {context} \n\n User question: {question}")
        ]
    )

    # Chain
    global llm_chain_fallback_with_context
    llm_chain_fallback_with_context = prompt | llm_fallback_with_context| StrOutputParser()
    #test_docs = retriever.vectorstore.similarity_search("getApprovals")
    ## Run
    #question = "write a code snipped that uses the getApprovals() method of the Smart Vaults class"
    #generation = llm_chain.invoke({"context": test_docs, "question": question})
    #if should_print_logs: print(generation)
    return llm_chain_fallback_with_context

def init_retrieval_grader():
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    global retrieval_grader 
    retrieval_grader= grade_prompt | structured_llm_grader
    #question = "how to create a vault"
    #docs = retriever.get_relevant_documents(question)
    #doc_txt = docs[0].page_content
    #if should_print_logs: print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
    return retrieval_grader

def init_generate():
   # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


    # Chain
    global generate_chain 
    generate_chain = prompt | llm | StrOutputParser()

    ## Run
    #generation = rag_chain.invoke({"context": docs, "question": question})
    #if should_print_logs: print(generation) 
    return generate_chain



def init_hallucination_grader():
    ### Hallucination Grader

    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    global hallucination_grader 
    hallucination_grader = hallucination_prompt | structured_llm_grader
   # hallucination_grader.invoke({"documents": docs, "generation": generation})
    return hallucination_grader

def init_answer_grader():
    # Data model
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    global answer_grader 
    answer_grader = answer_prompt | structured_llm_grader
    # answer_grader.invoke({"question": question,"generation": generation})
    return answer_grader

def init_question_rewriter():
    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n
        for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )

    global question_rewriter
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    #question_rewriter.invoke({"question": question})
    return question_rewriter
    

def init_router():
    # Data model
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["vectorstore", "web_search", "llm_fallback", "llm_fallback_with_context"] = Field(
            ...,
            description="Given a user question return the correct datasource.",
        )


    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    # structured_llm_router = llm.bind_tools([web_search, vectorstore])

    # Prompt
    system = """You are an expert at routing a user question to a vectorstore, a web_search, a llm_fallback_with_context, or a llm_fallback.
    The vectorstore is only for questions or requests that are related to the topics: Smart Vaults.
    The web_search is only for questions or requests that are related to news or that are probably referring to recent events.
    The llm_fallback_with_context is for any question or request that necessarily requires (or builds on) the context provided by the vectorstore or web_search, for example:
        - How will you improve the getApprovals method of the Smart Vaults class?
        - Write a code snipped that uses the Smart Vaults methods to create a 5 of 5 multisig proposal
    The llm_fallback_with_context superseeds the vectorstore and web_search.
    The llm_fallback is for any question or requests that do not fall in the categories above.
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    global question_router 
    question_router = route_prompt | structured_llm_router
    #response = question_router.invoke({"question": "What is the next weeks forecast for Ohio?"})
    #if should_print_logs: print(response)
    #response = question_router.invoke({"question": "What are the parameters of the getApproval method of the Smart Vault class?"})
    #if should_print_logs: print(response)
    #response = question_router.invoke({"question": "How will you improve the getApprovals method of the Smart Vaults class?"})
    #if should_print_logs: print(response)
    #response = question_router.invoke({"question": "How is the Jacobian operator defined?"})
    #if should_print_logs: print(response)
    return question_router

def init_web_search():
    global web_search_tool
    web_search_tool = TavilySearchResults(k=3)

def build_index():

    print("Building index...")

    ### from langchain_cohere import CohereEmbeddings

    # Set embeddings
    embd = OpenAIEmbeddings()

    # Docs to index
    urls = settings_manager.get_setting('urls')

    # Load
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split
    #text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #    chunk_size=500, chunk_overlap=20
    #)

    # Split your website into big chunks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 * 4, chunk_overlap=0)

    # This text splitter is used to create the child documents. They should be small chunk size.
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=125*4)

    #text_splitter = RecursiveCharacterTextSplitter(
    #    chunk_size=1000,
    #    chunk_overlap=200,
    #    length_function=len,
    #    is_separator_regex=False,
    #)

    doc_splits = parent_splitter.split_documents(docs_list)

    # Add to vectorstore
    vectorstore = Chroma(
        #documents=doc_splits,
        collection_name="parent_document_splits",
        #embedding=embd,
        embedding_function=OpenAIEmbeddings()
    )
    # The storage layer for the parent documents
    docstore = InMemoryStore()


    # retriever = vectorstore.as_retriever()

    global retriever 
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(doc_splits)
    #num_parent_docs = len(retriever.docstore.store.items())
    #num_child_docs = len(set(retriever.vectorstore.get()['documents']))
    #print (f"You have {num_parent_docs} parent docs and {num_child_docs} child docs")
    print("Index successfully built.")
    return retriever

## HELPERS

def init_all():
    build_index()
    init_router()
    init_llm_fallback()
    init_llm_fallback_with_context()
    init_retrieval_grader()
    init_generate()
    init_hallucination_grader()
    init_answer_grader()
    init_question_rewriter()
    init_web_search()
    build_graph()
    print("Done.")

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    if should_print_logs: print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    if should_print_logs: print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = generate_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    if should_print_logs: print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            if should_print_logs: print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            if should_print_logs: print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    if should_print_logs: print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})

    if should_print_logs: print(f"Original question: {question}")
    if should_print_logs: print(f"Better question: {better_question}")
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    if should_print_logs: print("---WEB SEARCH---")

    response = input(f"Your question requires a web search, would you like to continue? (yes/no): ")
    if response.lower() != 'yes':
        return 
    
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

def llm_fallback_with_context(state):
    """
    Generate answer using the LLM with vectorstore or web search.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    if should_print_logs: print("---LLM Fallback With Context---")
    state = retrieve(state)
    state = grade_documents(state)
    should_generate = decide_to_generate(state)
    if should_generate == "generate":
        state = generate(state)
    elif should_generate == "not supported":
        llm_fallback(state)
    question = state["question"]
    context = state["documents"]
    generation = llm_chain_fallback_with_context.invoke({ "context" : context, "question": question})
    return {"question": question, "generation": generation}

def llm_fallback(state):
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    if should_print_logs: print("---LLM Fallback---")
    question = state["question"]
    generation = llm_chain_fallback.invoke({"question": question})
    return {"question": question, "generation": generation}


### Edges ###

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    if should_print_logs: print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})

       # Fallback to LLM or raise error if no decision
    if source.datasource == 'llm_fallback_with_context':
        if should_print_logs: print("---ROUTE QUESTION TO LLM WITH CONTEXT---")
        return "llm_fallback_with_context"
    if source.datasource == 'llm_fallback':
        if should_print_logs: print("---ROUTE QUESTION TO LLM---")
        return "llm_fallback"
    if source.datasource == 'web_search':
        if should_print_logs: print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == 'vectorstore':
        if should_print_logs: print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    grade_docs = settings_manager.get_setting('grade_docs')
    if grade_docs == 'no':
        if should_print_logs: print("---DECISION: GENERATE (DOCUMENT GRADING TURNED OFF)")
        return "generate"
    if should_print_logs: print("---GRADING DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        if should_print_logs: print("---DECISION: ALL DOCUMENTS ARE IRRELEVANT TO QUESTION, MAYBE TRANSFORMING THE QUERY COULD HELP---")
        should_transform = input(f"All of the retrived documents were graded irrelevant for the question. Would you like to transform the query and try again? (yes/no): ")
        if should_transform.lower() == 'yes':
            return "transform_query"
        return "not supported"
    else:
        # We have relevant documents, so generate answer
        if should_print_logs: print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    should_check_hallucination = settings_manager.get_setting('check_for_hallucinations')
    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    if should_check_hallucination == 'yes':

        if should_print_logs: print("---CHECKING FOR POSSIBLE HALLUCINATIONS---")
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score
    else:
        if should_print_logs: print("--- HALLUCINATION CHECK TURNED OFF, SKIPPING---")
        grade = "yes"

    if grade == "yes":
        # Check question-answering
        should_check_for_inconsistencies = settings_manager.get_setting('check_for_inconsistencies')
        if should_check_for_inconsistencies == 'no':
            if should_print_logs: print("---QUESTION-ANSWERING CHECK TURNED OFF, SKIPPING---")
            return "useful"
        if should_print_logs: print("---CHECKING IF GENERATION REALLY ADDRESSES QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        if grade == "yes":
            if should_print_logs: print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            if should_print_logs: print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            should_transform = input(f"Generation doesn't address question. Would you like to transform the query and try again? (yes/no): ")
            if should_transform.lower() == 'yes':
                return "transform_query"
            else:
                return "not supported"
    else:
        if should_print_logs: print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        should_transform = input(f"Generation is not grounded in documents. Would you like to transform the query and try again? (yes/no): ")
        if should_transform.lower() == 'yes':
            return "transform_query"
        
        return "not supported"
    

def build_graph():

    print("Building graph...")
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """
        question : str
        generation : str
        documents : List[str]

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search", web_search) # web search
    workflow.add_node("retrieve", retrieve) # retrieve
    workflow.add_node("llm_fallback", llm_fallback) # llm fallback
    workflow.add_node("llm_fallback_with_context", llm_fallback_with_context) # llm_fallback_with_context
    workflow.add_node("grade_documents", grade_documents) # grade documents
    workflow.add_node("generate", generate) # generatae
    workflow.add_node("transform_query", transform_query) # transform_query

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
            "llm_fallback": "llm_fallback",
            "llm_fallback_with_context": "llm_fallback_with_context",
        },
    )
    workflow.add_edge("llm_fallback_with_context", END)
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
            "not supported": END,
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": END,
            "useful": END,
            "transform_query": "transform_query",
        },
    )
    workflow.add_edge("llm_fallback", END)

    # Compile
    global app
    app = workflow.compile()
    print("Graph successfully built.")

    return app

def ask_question(question):
    ducks = ''' \n RubberDuckAssistant says: 
    <(.)__
    (___ />> \n '''
    inputs = {"question": question}

    should_print_logs = settings_manager.get_setting('print_logs') == 'yes'
    
    if should_print_logs: 
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pif should_print_logs: print(value["keys"], indent=2, width=80, depth=None)
            if should_print_logs: print("\n")
    else:
        print("Thinking...")
        value = app.invoke(inputs)

    print(ducks)
    # Final generation
    pprint(value["generation"], indent=4 , width=100)

class SettingsManager:
    def __init__(self, filename):
        self.filename = filename
        self.load_settings()

    def load_settings(self):
        try:
            with open(self.filename, 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            self.settings = {}

    def print_enumerated_settings(self):
        for i, (key, value) in enumerate(self.settings.items()):
            if key == 'urls':
                print(f"{i+1}. URLs to index: \n")
                for i,url in enumerate(value):
                    print(f"    {i+1}. {url} \n")
            else:
                print(f"{i+1}. {key}: {value}")

    def change_setting_by_index(self, index):
        settings_list = list(self.settings.items())
        key = settings_list[index][0]
        if key == 'urls':
            new_value = maybe_ask_for_urls()
            self.settings[key] = new_value
            self.save_settings()
            print(f"Setting '{key}' updated to '{self.settings[key]}'")
            build_index()
        else:
            value = input(f"Enter the new value for {key}: ")
            is_valid = value.lower() == 'yes' or value.lower() == 'no'
            if not is_valid:
                while True:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    value = input(f"Enter the new value for {key}: ")
                    if value.lower() == 'yes' or value.lower() == 'no':
                        break
            self.settings[key] = value
            self.save_settings()
            print(f"Setting '{key}' updated to '{self.settings[key]}'")

    def save_settings(self):
        with open(self.filename, 'w') as f:
            json.dump(self.settings, f, indent=4)

    def get_setting(self, key):
        return self.settings.get(key)

    def set_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()

    def delete_setting(self, key):
        del self.settings[key]
        self.save_settings()

settings_manager = SettingsManager('settings.json')

def maybe_ask_for_urls():
    urls = settings_manager.get_setting('urls')
    if len(urls) == 0 or urls is None:
        urls = input("Enter the URLs to index, separated by commas: ")
    else:
        maybe_change_urls = input(f"URLs to index already set to: {urls}. Would you like to change them? (yes/no): ")
        if maybe_change_urls.lower() == 'yes':
            add_or_delete_urls = input("Would you like to add or delete URLs? (add/delete): ")
            if add_or_delete_urls.lower() == 'add':
                new_urls = input("Enter the URLs to add, if multiple, separate by commas: ")
                new_urls_list = new_urls.split(',')
                urls = urls + new_urls_list
            elif add_or_delete_urls.lower() == 'delete':
                urls = input("Enter the new URLs, if multiple, separate by commas: ")

    settings_manager.set_setting('urls', urls)
    return urls

def confirm_settings():
    print("Current settings:")
    settings_manager.print_enumerated_settings()
    confirm = input("Would you like to change these settings? (yes/no): ")
    is_valid_input = confirm.lower() == 'yes' or confirm.lower() == 'no'
    if not is_valid_input:  
        while True:
            print("Invalid input. Please enter 'yes' or 'no'.")
            confirm = input("Would you like to change these settings? (yes/no): ")
            if confirm.lower() == 'yes' or confirm.lower() == 'no':
                break
    if confirm.lower() == 'yes':
        while True:
            setting_to_change = input("Enter the number of the setting you would like to change: ")
            is_number = setting_to_change.isdigit()
            if not is_number:
                print("Invalid input. Please enter a valid setting number.")
                continue
            setting_to_change = int(setting_to_change)
            is_valid = 0 < setting_to_change <= len(settings_manager.settings)
            if not is_valid:
                print("Invalid setting number. Please enter a valid setting number.")
                continue
            settings_manager.change_setting_by_index(setting_to_change - 1)
            maybe_continue = input("Would you like to change another setting? (yes/no): ")
            is_valid = maybe_continue.lower() == 'yes' or maybe_continue.lower() == 'no'
            if not is_valid:
                while True:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    maybe_continue = input("Would you like to change another setting? (yes/no): ")
                    if maybe_continue.lower() == 'yes' or maybe_continue.lower() == 'no':
                        break
            if maybe_continue.lower() == 'no':
                break
    return True

class RubberDuckAssitantCLI(cmd.Cmd):
    intro = 'Welcome to Rubber Duck. Type "help" for available commands or just start asking questions!.'
    prompt = 'Ask a question > '
    def __init__(self):
        super().__init__()

    def default(self, line):
        """Start session."""
        question = line
        if question == 'quit' or question == 'exit':
            return True
        ask_question(question)

    def do_quit(self, line):
        """Exit the CLI."""
        return True

    def do_change_settings(self, line):
        """Change settings."""
        confirm_settings()


    def postcmd(self, stop, line):
        print() 
        return stop

    def preloop(self):
        # Add custom initialization here
        confirm_settings()
        print("Loading tools...")
        init_all()


## HELPERS 



if __name__ == '__main__':
    RubberDuckAssitantCLI().cmdloop()
