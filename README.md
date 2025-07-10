---
license: mit
task_categories:
- feature-extraction
- text-classification
- sentence-similarity
- question-answering
language:
- en
tags:
- Embeddings
- India
- Supreme
- Court
- Legal
size_categories:
- n<1K
---



# Indian Supreme Court Judgements Chunked


## Executive Summary
The dataset aims to address the chronic backlog in the Indian judiciary system, particularly in the Supreme Court, by creating a dataset optimized for legal language models (LLMs). The dataset will consist of pre-processed, chunked, and embedded textual data derived from the Supreme Court's judgment PDFs.

### Problem and Importance - Motivation
Indian courts are overwhelmed with pending cases, with the average time to resolve cases in High Courts being 5.3 years and some Supreme Court cases dating back to 1982. Despite initiatives like the "eCourts Initiative," there remains a lack of digital, structured resources tailored for advanced computational tools like LLMs. Thus since there was relatively no conversation about the integration of advanced retrieval methods for Indian Law Documents, it was thought the conversation should be started here where the data is already publicly available. Chunking these documents would allow for people to easily embed them and start a retrieval process. This dataset was created as a proof of concept, with a power analysis included which highlights if the criteria to understand if this dataset makes a difference or not. 

### Proposed Solution
This project will process approximately 100-150 judgments from the Supreme Court of India, converting PDFs into structured text, and applying chunking and embedding strategies. The dataset will be made accessible through GitHub, with plans to include vectorized data for retrieval and generative applications. Tools such as Python, Chromadb, Langchain, and Pandas will be utilized for data processing and storage.

### Unique Contribution
The dataset will be the first of its kind to provide Indian legal judgments in a format ready for LLMs, differentiating it from existing datasets that either lack preprocessing or focus on metadata extraction only. This will enable applications in legal research, decision support, and document retrieval tailored to the Indian legal context.

### Potential Impact
This initiative could revolutionize legal workflows in India by reducing case backlog, enhancing judicial efficiency, and democratizing access to advanced legal technology. The dataset could serve as a foundational resource for AI-driven legal solutions, aligning with broader governmental digitalization goals.

### Solution Explained
This dataset consists of the original PDF documents chunked in various ways - Recursive, Semantic, and TokenWise. Older systems use archaic technologies such as keyword searches to obtain legal judgments that lawyers may use for their research. By creating this dataset of chunked judgments I aim to create a proof of concept that these judgments can be retrieved easier via newer technologies and methods such as Retrevial Augmented Generation. This can be done so that users can be presented with the most relevant set of documents in the most efficient amount of time possible. 

### Potential Applications
- Legal Firms with an interest in getting beyond its' competition can use this for better retrieval.
- Private companies can use it to check if embedding and retieval is a better idea that current keyword matching approaches.
- The Supreme Court of India can check to see if this method helps them in their retieval process.
- With the number of backlog cases in the Supreme Court of India, better retrieval methods to help in finding old cases to act as precedent or even to uncover facts might make the process much faster and better.

### Potential Biases
- The dataset contains only 100 samples as this is a proof of concept, but can be easily extended to how many ever samples are required, with the tool present in the github link.
- Some pages include few words as content on headers and footers, this was not removed. It should not effect the embeddings but incase it does this is some future work that we can work on.

### Review of Previous Datsets
 - There is another work which aims to annotate parts of a criminal bail application, and create a repository of annotated bail applications - link to the paper - https://link.springer.com/chapter/10.1007/978-981-99-9518-9_30
 - It did not solve the exact same problem but it was in line with the eCourts Initiative by the Government of India as well.
 - There is also another Kaggle dataset which has the data that the Supreme Court of India provides, but they have only extracted the metadata and given the rest of the data in PDF format. There has been no processing of the actual text. - https://www.kaggle.com/datasets/vangap/indian-supreme-court-judgments/data
 - Upon speaking to law students in India as well as the USA, it was learnt that there are a few more accessible datasets by LexisNexus but they are paid for.
 - The US link is here - https://www.lexisnexis.com/en-us
 - The link to the Indian Dataset is here - https://www.lexisnexis.co.in/en-in/home.page

### Tools Used for this Dataset
- Indian Supreme Court Judgment Data
- Python 3.12
- All requirements highlighted in the github link associated with this repository below
- Open AI API


## Description of data
- Original Judgements folder - This contains the original judgments in PDF format, numbered 1 to 100.
- Normal Text Folder - Consists of the data in PDF format converted to normal text including line breaks. This is a literal conversion from PDF to text without preprocessing. It has been included to allow users to experiment further.
- Text Folder - Consists of the data in PDF format converted to text excluding line breaks and after removing special characters. It has been included to allow users to experiment further.
- Recursive Folder - Consists of the original judgments chunked using recursive techniques. It uses the RecursiveCharacterTextSplitter from langchain, where chunk size is defined as 1000 and overlap is 200. This technique is meant for experimentation of larger chunks with more context and larger overlap. Chunks are separated with "---" characters on a new line.
- TokenWise Folder - Consists of the original judgments chunked using token-wise techniques. It uses the TokenTextSplitter from Langchain, where chunk size is defined as 100 and overlap is 20. This technique is meant for the experimentation of smaller chunks with less context and smaller overlap. Chunks are separated with "---" characters on a new line.
- Semantic Folder - Consists of the original judgments chunked using semantic techniques. It uses the SemanticChunker from langchain, which uses the help of the OpenAIEmbeddings. This method of chunking aims to chunk data based on the topic of information in it. With the help of OpenAIEmbeddings, topics are extracted from the text, and whenever there is a change of topic a new chunk is created. This method was used to help in experimenting on whether chunks with more context-specific information will assist in retrieval. Chunks are separated with "---" characters on a new line.
- metadata Folder - Consists of the metadata of the original judgments, this can be used to identify the metadata or case specifics with the the retrieved embeddings.
Each folder consists of data that has been suffixed with an integer value. This value remains the same across all folders, for easier access and to relate data from two different folders with each other.

## Power analysis results
Effect Size: 0.5 (Moderate) - Assuming
Calculated using Cohen’s D,
Justification - We are using a medium effect size, as in this case we would not like to be very conservative by choosing a 0.2, and neither do we want to choose a large effect size of 0.8.
Hence this number was chosen at a medium level of 0.5.

Significance Level (α): 0.05 - Common Value

Power: 0.8 - Traditionally Set

Power Analysis Test used - statsmodels.stats.power.TTestPower

Sample Size Needed = 33.367

Approximately Equal to - 34

This Database will need approximately 34 queries run on it to determine if it is more useful than
traditional systems such as LexisNexis for legal document case retrieval.

## Exploratory Data Analysis
1. Recursive
   - Average chunk size: 981.46 characters
   - Average words per chunk: 92.90
   - Top 10 most occurring words:
     court: 4174,
     section: 2215,
     order: 1620,
     case: 1613,
     appeal: 1612,
     appellant: 1505,
     act: 1425,
     high: 1416,
     accused: 1232,
     also: 1114
   - Total unique words: 19658
   - Average word length: 6.33 characters
   - Chunk length distribution:
     Min: 201,
     Max: 1000,
     Median: 996.00,
     Standard deviation: 87.26
   - Word count distribution:
     Min: 23,
     Max: 158,
     Median: 92.00,
     Standard deviation: 12.32,
   - Percentage of chunks with numbers: 98.94%
   - Average sentences per chunk: 8.15
   - Token frequency distribution:
     Tokens appearing only once: 7352,
     Tokens appearing 2-5 times: 7110,
     Tokens appearing 6-10 times: 1740,
     Tokens appearing more than 10 times: 3456

2. Semantic
   - Average chunk size: 2017.98 characters
   - Average words per chunk: 191.13
   - Top 10 most occurring words:
   court: 3404,
   section: 1780,
   order: 1332,
   appeal: 1325,
   case: 1289,
   appellant: 1215,
   high: 1162,
   act: 1131,
   accused: 982,
   dated: 895
   - Total unique words: 19658
   - Average word length: 6.33 characters
   - Chunk length distribution:
   Min: 2,
   Max: 16549,
   Median: 1121.00,
   Standard deviation: 2462.72
   - Word count distribution:
   Min: 0,
   Max: 1500,
   Median: 109.00,
   Standard deviation: 225.39
   - Percentage of chunks with numbers: 91.25%
   - Average sentences per chunk: 15.95
   - Token frequency distribution:
   Tokens appearing only once: 9680,
   Tokens appearing 2-5 times: 5441,
   Tokens appearing 6-10 times: 1523,
   Tokens appearing more than 10 times: 3014

3. Token-Wise

   - Average chunk size: 440.22 characters
   - Average words per chunk: 41.84
   - Top 10 most occurring words:
   court: 4234,
   section: 2226,
   order: 1659,
   appeal: 1638,
   case: 1595,
   appellant: 1509,
   high: 1427,
   act: 1407,
   accused: 1230,
   also: 1125,
   - Total unique words: 20455
   - Average word length: 6.31 characters
   - Chunk length distribution:
   Min: 57,
   Max: 664,
   Median: 452.00,
   Standard deviation: 69.88
   - Word count distribution:
   Min: 9,
   Max: 59,
   Median: 42.00,
   Standard deviation: 5.23
   - Percentage of chunks with numbers: 93.35%
   - Average sentences per chunk: 3.61
   - Token frequency distribution:
   Tokens appearing only once: 8282,
   Tokens appearing 2-5 times: 6953,
   Tokens appearing 6-10 times: 1722,
   Tokens appearing more than 10 times: 3498

## Link to publicly available data sourcing code repository
https://github.com/vihaannnn/Individual-Dataset

## Ethics Statement
- Data Privacy and Anonymization - All judgment data will be sourced from publicly accessible platforms, such as the Supreme Court of India's website. No personal or sensitive information about individuals involved in the cases will be included that is not already present on the Supreme Court of India’s website, ensuring compliance with privacy laws and ethical standards.
- All data was ethically sourced directly from the Supreme Court of India website and it was ensured that no private information which is outside of these documents were included.
- Responsible Use - The dataset is intended solely for research and technological advancements in legal applications. Any misuse, such as for unethical profiling or unauthorized commercial purposes, will be explicitly prohibited in the terms of use.
- Transparency and Reproducibility - The methods used for data collection, chunking, and embedding will be documented comprehensively to promote transparency. The dataset and code will be made publicly available through platforms like HuggingFace, enabling reproducibility and fostering open collaboration.
- Bias and Fairness - Care will be taken to ensure the dataset does not reinforce or introduce biases inherent in the source material. Regular audits will be conducted to identify and mitigate any potential biases in the processed data whenever the data is reloaded.
- Respect for Legal Frameworks - This project will strictly adhere to all applicable laws, including those governing intellectual property and access to government data. Efforts will align with the "eCourts Initiative," which promotes technology development for the Indian judiciary.
- Minimizing Harm - The project will prioritize minimizing harm by preventing data misuse and avoiding unintended consequences, such as misinterpretation of legal judgments due to incorrect data chunking or embeddings. No data has been added explicitly, all data in chunks have been mined from the original pdf documents.
- Code Details and Methods - The project employs several strategies to preprocess and chunk textual data, ensuring optimal structure for LLM applications:
  - Recursive Character Chunking: Uses the RecursiveCharacterTextSplitter with parameters (1000 characters per chunk, 200-character overlap) to create chunks while maintaining context.
  - Token-Wise Chunking: Implements the TokenTextSplitter (100 tokens per chunk, 20-token overlap) for fine-grained segmentation based on token count.
  - Semantic Chunking: Utilizes the SemanticChunker powered by OpenAI embeddings to split text into semantically coherent units.
  
  The process also includes cleaning text to remove invisible and non-standard characters, enhancing the quality and utility of the dataset.
- Automation and Transparency - The provided Python scripts automate the workflow, from extracting text from PDFs using pdfplumber to chunking with advanced text splitters. The source code will be made publicly available, ensuring transparency in data processing methods.
- Ethical Data Processing - The project uses publicly available Supreme Court judgment PDFs, with no modifications to original legal content. Preprocessing steps strictly remove hidden or extraneous characters without altering the legal meaning or structure.
- Data Integrity - By using semantic chunking and embedding methods, the dataset preserves the context and logical structure of legal judgments, ensuring that the processed data remains meaningful and accurate.
- Responsible Use and Sharing - All datasets and associated code will be shared under appropriate licenses - MIT License that prohibits misuse, including unethical profiling or discriminatory applications. The emphasis will be on research and development to assist in reducing court backlogs.
- Bias Mitigation and Fair Representation - The chunking algorithms are applied uniformly across all data, minimizing the risk of selective bias. Semantic processing aims to enhance data consistency and usability across diverse legal scenarios.


Ethics of the MIT License
- Freedom to Use, Modify, and Distribute
  - The MIT License allows anyone to use, modify, and distribute the licensed software, whether for private, commercial, or academic purposes.
  - Ethical Implication: This aligns with the principle of knowledge sharing and the democratization of technology, fostering innovation and collaboration.
- Attribution Requirement
  - The license requires users to include the original copyright notice and a copy of the license in distributed software.
  - Ethical Implication: This ensures proper credit is given to the original creators, recognizing their contributions and promoting transparency.
- No Liability or Warranty
  - The license explicitly disclaims warranties and liability, meaning users take full responsibility for how they use the software.
  - Ethical Implication: While this protects developers from legal risks, it shifts the responsibility to users, who must ethically consider the impact of their use of the software.
- Lack of Restrictions on Usage
  - The permissive nature of the MIT License allows the software to be incorporated into both open-source and proprietary projects.
  - Ethical Implication: This flexibility can lead to ethical dilemmas, such as the software being used for purposes the original developers might find objectionable (e.g., surveillance, weapons development). Developers using the MIT License should be aware of this possibility and decide whether they are comfortable with it.
- Promotion of Open Collaboration
  - The license encourages a culture of openness by removing barriers to adoption and modification.
  - Ethical Implication: This supports the global sharing of technology and ideas, benefiting both the tech community and society at large.

## Open Source License
MIT License
More info - https://choosealicense.com/licenses/mit/

## Data Collection Protocol

1. Data Sources:
- Primary Source: Supreme Court of India's official website for judgments: https://www.sci.gov.in/judgements-judgement-date/.
- Format: PDF files containing judgment texts.
2. Sampling Plan:
- Scope: A sample of 100 judgment documents based on the fact that this was a Proof of Concept and the power analysis required 34 queries to be successfully processed.
- Criteria: Documents selected will represent diverse case types to ensure comprehensive coverage, this was done by choosing all documents for particular date ranges ensuring that all types of documents were covered.
3. Data Collection Methods:
Tools:
- Python-based Program:
  - Reads raw PDF files from a designated folder.
  - Extracts textual content from PDFs using libraries like PDFplumber.
- Storage and Organization:
  - Metadata and chunked text stored in structured folders.
  - Text files organized by chunking strategies (e.g., by recursive, semantic, tokenwise).
4. Data Processing:
- Chunking Methods:
Divide the text into manageable segments via a python program for processing (recursive, semantic, tokenwise).
5. Ethical Considerations:
- Privacy: Ensure no sensitive or personally identifiable information that is not already open-sourced by the supreme court is included in the processed dataset.
- Accessibility: Data processed and uploaded to HuggingFace for open access in compliance with the eCourts Initiative of the Indian Judiciary.
6. Tools:
- Python3, Pandas, PDFplumber (for text extraction), LangChain (for chunking), OpenAI API (for semantic chunking), and everything in the requirements.txt file on the github repo.
7. Quality Assurance:
- Conduct a pilot test on a subset of documents to validate chunking and embedding methods.
- Perform manual reviews to ensure the accuracy of text extraction and chunking.
- Generate logs for all processing stages for traceability.
- Conduct unit test cases of code ensuring everything is running as required.
8. Data Management:
- Storage: Text files saved in structured directories with clear naming conventions.
- Versioning: Use GitHub for maintaining versions of text files and vector databases as well as branching.
9. Limitations:
- The sample size is constrained to avoid potential overfitting or hallucination in LLM applications as this is a proof of concept.


## Credits
- Original data sourced from - https://www.sci.gov.in/judgements-judgement-date/
- Parts of this README were generated using AI tools such as - ChatGPT, Perplexity, and Claude.