# AUTOMATED-DATA-MODELING-TOOLS

Automated Data Modeling tools is my first personal project to try out the application of the LangChain and LangGraph frameworks in software development.

I integrated LangChain with the LLM workflow to allow statistical tool requests and diagnostics. After users have uploaded their dataset, they can ask statistical questions about the dataset, including but not limited to correlation test, normality test, multicollinearity test, etc.

Next, I integrated a knowledge base as RAG for the LLM to retrieve guidance that helps automate the data modeling process through the application. The RAG contains the data modeling options and recommendations according to the EDA processes conducted by the app. Once auto mode is enabled, the LLM will recommend the most suitable data cleaning approach and data modeling techniques, which include feature selection and hyperparameter tuning. 

Users can also override the AI's selection via direct modification, or instruct through the chatbox provided in the left sidebar. If users prefer to conduct data modeling themselves, Auto AI mode can be turned off. Users can then conduct EDA through various summaries and visualization techniques. No worries, the AI assistant is still there to provide hints from time to time.   

