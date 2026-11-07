# Lets-Learn-LangChain
LangChain provides a structured way to connect LLMs with external data, tools, databases, APIs, and other components to build intelligent applications. Consider a normal LLM application.


### LangChain – Complete Conceptual Notes

These notes are designed to help you understand LangChain from an AI Engineer's perspective, not as a programming tutorial.

Our focus will be on:

* Understanding the core concepts and terminology.

* Understanding why LangChain exists and what problems it solves.

* Understanding how LLMs, prompts, chains, tools, agents, and memory work together.

* Learning the architecture of real-world LLM applications.

* Preparing for AI Engineer interviews and technical discussions.

### 1\. Introduction to LangChain

### 1.1 What is LangChain?

LangChain is an open-source framework used to develop applications powered by Large Language Models (LLMs).

In simple words:

LangChain provides a structured way to connect LLMs with external data, tools, databases, APIs, and other components to build intelligent applications.

Consider a normal LLM application.

You ask ChatGPT:

"What is machine learning?"

The LLM generates an answer using its learned knowledge.

But imagine building an application that can:

* Read your company's PDF documents.

* Search information from the internet.

* Retrieve customer information from a database.

* Perform mathematical calculations.

* Send emails.

* Generate answers based on private company data.

A standalone LLM cannot necessarily perform all these operations by itself.

This is where LangChain becomes useful.

![LangChain: Designing Real-World LLM Applications | by Koduru Meghana | Apr, 2026 | Medium](https://images.openai.com/static-rsc-4/zIy_Lb7I2G8V9INeE7oFITVcDCweOP9qSKBDANnkluaat7rr_dF-aUkaG-TTa3-zJ0sc6hR8OxsIfVlYbz0waLltrODD5hCC9JvpYhi0MPPw1HWqRN1HsswXZfbq62tuI-vUxUgGpSb2JP4uz0Y1n4lTQTiUf00C9xQZ51OeF7SK-l8jiUSGqYGNjLx66Pzc?purpose=fullsize)

### 1.2 Why do we need LangChain?

An LLM has several limitations when used independently.

| LLM limitation                                    | How LangChain helps                                   |
| ------------------------------------------------- | ----------------------------------------------------- |
| Cannot automatically access private documents     | Connects document loaders and retrievers              |
| Cannot independently call external APIs           | Integrates tools                                      |
| Limited knowledge of recent information           | Connects search and retrieval systems                 |
| Cannot automatically execute multi-step workflows | Supports chains and agents                            |
| No guaranteed access to conversation history      | Integrates conversation-history mechanisms            |
| Cannot directly interact with every database      | Provides integrations with databases and data sources |

Important: LangChain does not make an LLM inherently more intelligent. It provides the infrastructure to build applications around it.

### 1.3 LangChain vs LLM

These two terms are not interchangeable.

| LLM                                | LangChain                                       |
| ---------------------------------- | ----------------------------------------------- |
| Language model                     | Application development framework               |
| Generates and understands language | Orchestrates application components             |
| Examples: GPT, Claude, Gemini      | Framework for connecting models, tools and data |
| Performs inference                 | Helps build complete LLM workflows              |

Remember:

LLM = Brain

LangChain = System that connects the brain with the outside world.

### 2\. Core Architecture of LangChain

Before studying individual components, understand the overall architecture.

A typical LangChain application consists of these major components:

![Lang Chain Deep Dive: Building Modular LLM Applications from Scratch | by Siddhantdeore | Apr, 2026 | Medium](https://images.openai.com/static-rsc-4/mkm3KVg2xPHpQruwn0UK1TXlg_yPtWfcfKMWeGzuvwW-oL-0pO9Fyur2RuF6GQcVMza0Ma7ygnhJ6gZ86jeU2RKA55yjlLQ9ICzFFh_iBrkZUfnU5lgHyt_I4V03Zl3qeThmv_EDfA1ydNWLGwErSHqe4A-LvaR9gr9d7c_V8WaanUBjNT-jkLYVMrhLiU2q?purpose=fullsize)

* Models: The intelligence layer that processes input and generates output.

* Prompts: Instructions and context provided to the model.

* Output Parsers: Convert model responses into structured formats.

* Retrievers: Retrieve relevant information from external knowledge sources.

* Tools: Allow the model to interact with external systems.

* Chains: Connect multiple operations into a workflow.

* Agents: Decide which actions or tools to use to accomplish a task.

* Conversation History: Provides context from previous interactions.

Let's understand each component in detail.

### 3\. Models in LangChain

Models are the central components of an LLM-powered application.

LangChain supports different types of models.

### 3.1 Language Models

A language model processes text and generates text.

For example:

Input: "Explain artificial intelligence."

Output: "Artificial intelligence is a branch of computer science..."

Traditional language-model interfaces and chat-model interfaces are conceptually different.

### 3.2 Chat Models

Chat models are designed for conversational interactions.

They work with structured messages rather than treating everything as one simple text string.

Three important message roles are:

| Message type | Purpose                                        |
| ------------ | ---------------------------------------------- |
| System       | Defines the model's behaviour and instructions |
| Human        | Contains the user's message                    |
| AI           | Contains the assistant's response              |

Example:

* System: You are an experienced mathematics teacher.

* Human: Explain probability.

* AI: Probability is the measure of how likely an event is to occur.

### 3.3 Popular models used with LangChain

* OpenAI GPT models

* Google Gemini models

* Anthropic Claude models

* Meta Llama models

* Mistral models

LangChain provides integrations that allow developers to work with different model providers through a common application framework.

### 3.4 Embedding Models

This is an especially important concept for AI Engineers.

An embedding model converts text or other supported data into numerical vectors.

For example:

"Artificial Intelligence"

may be represented as a vector containing hundreds or thousands of numerical dimensions.

Why is this useful?

Because computers can compare vectors to identify semantic similarity.

Consider:

* Sentence A: "How can I learn artificial intelligence?"

* Sentence B: "What is the best way to study AI?"

The wording is different, but their meanings are similar.

Embedding models help represent that semantic relationship.

Applications:

* Semantic search.

* Document retrieval.

* Recommendation systems.

* Retrieval-Augmented Generation (RAG).

* Similarity detection.

![Embeddings and semantic search](https://images.openai.com/static-rsc-4/rCMcmO3BFN5AefqZXox3KW2qI92gHb_VLiFD1JDlddn0XdJSOLXOnHAU30SpSRt7dcwV3uoBeEODllW86bkY_I_AsSc225H6bIswDgG3tjJw_MHAh15zp5M-NfBSduL2ONpMPcVsb27VUwevVncK0sGtb5b05hgulrVKejNVfooONVHm3CD--VTQhal-VUsF?purpose=fullsize)

Interview point: An LLM generates language, whereas an embedding model converts information into numerical representations that can be compared mathematically.

### 4\. Prompt Templates

### 4.1 What is a prompt?

A prompt is the instruction or input given to an LLM to generate a response.

Example:

"Explain machine learning in simple language."

However, in real-world applications, prompts often need to be dynamic.

For example, a teaching application may need to explain different subjects to different students.

Instead of manually creating a new prompt every time, we use prompt templates.

### 4.2 What is a Prompt Template?

A Prompt Template is a reusable structure containing predefined instructions and variable placeholders.

Example concept:

* Role: You are an experienced teacher.

* Topic: \[Topic\]

* Difficulty: \[Level\]

* Output: Explain with examples.

The application can replace the placeholders with different values.

For example:

| Variable | Value                     |
| -------- | ------------------------- |
| Topic    | Machine Learning          |
| Level    | Beginner                  |
| Output   | Explanation with examples |

The model receives a customized prompt.

### 4.3 Types of prompts

1\. String Prompt Template

A template designed around a single text string.

2\. Chat Prompt Template

A template designed around structured conversational messages such as system, human, and AI messages.

3\. Few-shot Prompt Template

A prompt that provides examples to guide the model's response.

Example:

Question: What is 2 + 2?

Answer: 4

Question: What is 3 + 3?

Expected answer: 6

The model learns the expected response pattern from the examples included in the prompt.

Remember: Prompt templates improve consistency, reusability, and control over LLM interactions.

### 5\. Output Parsers

### 5.1 What is an Output Parser?

An Output Parser is a component responsible for converting an LLM's response into a specific, usable format.

By default, an LLM generally produces natural-language text.

However, real-world applications often require structured outputs.

For example, imagine an AI recruitment system.

The LLM generates:

"Candidate has 3 years of Python experience, knows SQL, and has worked with machine learning."

But the application needs structured information:

| Field            | Extracted value |
| ---------------- | --------------- |
| Experience       | 3 years         |
| Python           | Yes             |
| SQL              | Yes             |
| Machine Learning | Yes             |

An Output Parser helps convert the model's response into the required structure.

### 5.2 Types of Output Parsers

1\. String Output Parser

Returns the model's response as plain text.

2\. JSON Output Parser

Helps produce structured JSON output.

3\. Structured Output Parser

Helps represent responses according to predefined fields and expected data types.

4\. List Output Parser

Helps convert responses into lists of individual items.

### 5.3 Why are Output Parsers important?

* Make model responses easier to process.

* Improve consistency in application outputs.

* Support integration with databases and APIs.

* Help build structured AI applications.

* Reduce the need for manual response processing.

Real-world example: An AI invoice-processing system extracts invoice number, date, vendor name, and total amount into structured fields.

### 6\. Documents and Document Loaders

### 6.1 What is a Document in LangChain?

A Document is a standard data structure used to represent a piece of information.

Conceptually, a Document contains two important elements:

* Page Content: The actual text or information.

* Metadata: Additional information about the content.

Example:

Imagine a company has a PDF containing employee policies.

A document may contain:

| Component           | Example                                 |
| ------------------- | --------------------------------------- |
| Page Content        | Employees receive 20 annual leave days. |
| Source metadata     | employee_policy.pdf                     |
| Page metadata       | Page 5                                  |
| Department metadata | Human Resources                         |

Metadata helps identify the origin and context of information.

### 6.2 What are Document Loaders?

Document Loaders are components that import information from different sources into an application.

Imagine building an AI assistant for a university.

The university has information stored in:

* PDF documents.

* Word documents.

* Excel spreadsheets.

* Websites.

* Text files.

* Databases.

Document Loaders help bring this information into a common document representation.

![Unleash Your LLM’s Knowledge: A Deep Dive into LangChain Document Loaders | by Himanshu Bisht | Medium](https://images.openai.com/static-rsc-4/sbrE6u0zG657dw31k37UQI3zB927VEa61aSNXTaHIirCc-52wu69rFOdbWCdZhATF6q8Wkg-yA9O-KiU8ShUtPkF22uNVVORyLLIgd22ho5ou3Aml-uPObsNrczUGXGwfHHtHGlVFVgDHgyfI_PuKM6xrYmZS0uu6yBsmRoV_AAuZyV_OZlN0NZMmoR5kZBI?purpose=fullsize)

### 6.3 Common types of Document Loaders

| Loader category  | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| PDF Loader       | Extracts content from PDF files                |
| CSV Loader       | Loads tabular CSV information                  |
| JSON Loader      | Loads JSON data                                |
| Web Loader       | Extracts content from websites                 |
| Directory Loader | Loads multiple files from a directory          |
| Database Loader  | Retrieves information from supported databases |

### 6.4 Why are Document Loaders important?

Without document ingestion, an AI application may not have access to the external information it needs.

Document Loaders create the initial connection between external knowledge sources and the AI application.

Remember: Document Loader = Bringing external information into the application.

### 7\. Text Splitters

### 7.1 What is Text Splitting?

Large documents can contain thousands or millions of words.

Sending an entire document to an LLM is often inefficient or impossible because models have context-window limitations.

Text Splitters divide large documents into smaller pieces called chunks.

Example:

Suppose a PDF contains 100 pages.

Instead of processing the entire PDF as one large document, the application divides it into smaller chunks.

![RAG Chunking Strategies: How to Split Text for Better Retrieval](https://images.openai.com/static-rsc-4/UnoucxtH2qPwL8L3c6oyHo1YINdJAB2j-NAJclW79fwHIsy9F0SklkTTuZ6OMbrDVu4PJ8KdQeF8PdYNQTVmCBIIJGFr25zXno7NUFqdbg_z3TTGMCkmP1U8eGMhrVQumU2yWfDMk2gIqOisQJdV2Jv_zPX2sQ49_2QDpAqjueBL6Fx5ZOGx2CguRVSqQPAL?purpose=fullsize)

### 7.2 What is Chunking?

Chunking is the process of dividing large content into manageable sections.

Example:

Original content:

"Artificial intelligence is a field of computer science. Machine learning is a subset of artificial intelligence. Deep learning is a subset of machine learning."

After chunking:

* Chunk 1: Artificial intelligence is a field of computer science.

* Chunk 2: Machine learning is a subset of artificial intelligence.

* Chunk 3: Deep learning is a subset of machine learning.

Each chunk can be processed and retrieved independently.

### 7.3 Important Text-Splitting Concepts

1\. Chunk Size

The maximum or target size of each chunk.

A chunk may be measured in characters, tokens, or another unit, depending on the splitter.

2\. Chunk Overlap

The amount of content shared between consecutive chunks.

For example:

* Chunk 1: AI is a field of computer science. Machine learning is a subset.

* Chunk 2: Machine learning is a subset. Deep learning is a subset of ML.

The repeated sentence creates overlap.

3\. Separators

The boundaries used to divide content.

Examples include:

* Paragraphs.

* New lines.

* Sentences.

* Words.

* Characters.

### 7.4 Why is Chunk Overlap important?

Consider a sentence that begins at the end of one chunk and continues into the next.

Without overlap, some contextual relationships may be lost.

Overlap helps preserve surrounding information.

However, excessive overlap increases storage and processing costs and can create redundant retrieval results.

### 7.5 Types of Text Splitters

* Character Text Splitter.

* Recursive Character Text Splitter.

* Token-based Text Splitter.

* Markdown Text Splitter.

* HTML Text Splitter.

* Language-specific Text Splitter.

Important interview concept: Recursive Character Text Splitter attempts to divide content using a hierarchy of separators, generally preserving larger semantic units where possible.

### 8\. Vector Databases and Vector Stores

This is one of the most important concepts in modern AI engineering.

### 8.1 What is a Vector?

A vector is an ordered collection of numerical values.

For example:

V = \[0.21,\\ 0.45,\\ -0.18,\\ 0.92\]

In AI applications, embeddings represent the semantic characteristics of information as vectors.

### 8.2 What is a Vector Database?

A Vector Database is a specialized database designed to store, index, and retrieve vector representations efficiently.

Imagine a university has 10,000 PDF documents.

A student asks:

"What is the eligibility for MCA admission?"

A traditional keyword search might look for the exact word "eligibility".

A semantic search system can retrieve relevant information even when the wording differs, such as:

"What are the admission requirements for MCA?"

This is possible because vector search compares semantic representations.

![🚀 Introduction to Vector Search | Sciences 44](https://images.openai.com/static-rsc-4/OifZLnUbHq_K-bK7KH0bcov4u9YQ-yw1Bfjfai51vxaQLT5q-jF_Tg2IpEUO37NMbEiaBBgG9RDmlGkLIQC0Z8YykVPsLXYEW0whSQQuFcyoyHmYrrEnYvlhWINd-pGxRyEQUBnJ6IQHdJddFnszyuJBspoycXE9kLK32yqBjtAzHRjkay5BmivROCM0UMNa?purpose=fullsize)

### 8.3 How does Vector Search work?

The conceptual process:

* Convert documents into embeddings.

* Store those embeddings in a vector database.

* Convert the user's query into an embedding.

* Compare the query vector with stored vectors.

* Retrieve the most relevant documents or chunks.

### 8.4 Similarity Search

Similarity Search identifies vectors that are mathematically close to one another.

Common similarity measures include:

1\. Cosine Similarity

Measures the cosine of the angle between two vectors.

It is commonly used to compare the directional similarity of embeddings.

2\. Euclidean Distance

Measures the straight-line distance between two vectors.

3\. Dot Product

Measures the product of corresponding vector dimensions and is commonly used for vector similarity calculations.

### 8.5 Popular Vector Databases

| Vector database | General purpose                           |
| --------------- | ----------------------------------------- |
| FAISS           | Efficient similarity search               |
| Chroma          | Vector storage and retrieval              |
| Pinecone        | Managed vector database                   |
| Milvus          | Scalable vector database                  |
| Weaviate        | Vector search and knowledge retrieval     |
| Qdrant          | Vector search with filtering capabilities |

### 8.6 What is a Vector Store in LangChain?

A Vector Store is an abstraction that allows LangChain applications to interact with vector storage systems.

It provides common operations such as:

* Adding documents.

* Storing embeddings.

* Performing similarity searches.

* Retrieving relevant documents.

Important distinction:

Embedding Model = Converts information into vectors.

Vector Database = Stores and searches vectors.

Vector Store Integration = Connects vector storage functionality to the LangChain application.

### 9\. Retrievers

### 9.1 What is a Retriever?

A Retriever is a component that retrieves relevant information from a knowledge source based on a query.

In simple language:

Retriever = Information Finder.

Consider a company chatbot.

An employee asks:

"What is the company's maternity leave policy?"

The retriever searches the available company documents and identifies the most relevant information.

It then returns the retrieved documents or document chunks.

![LangChain Framework and Core Components | Henry Hoang](https://images.openai.com/static-rsc-4/99km_8I-j5l_fsaOmOmfM5T_Bu5W09msWRtHL-zj94NCFYcdOe1PzNNw6osWTTHAc9AQm-RX66HOk0tRg2qRSPZEJdlICmNpQKk3SZqFaTfL5cMGX6mi6Dhw3zOYfdhcIbb_1JoMH6KxrqsxsvKJ8bRk0tg0Xx3f3fSSPFtCxmphw3XAo8IiTWQAfwV70qR9?purpose=fullsize)

### 9.2 Retriever vs Vector Store

These concepts are related but different.

| Vector Store                               | Retriever                              |
| -----------------------
