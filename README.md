# AUTOMATED-DATA-MODELING-TOOLS

Automated Data Modeling tools is my first personal project to try out the application of LangChain and LangGraph framework in software development.

I integrated LangChain in the LLM to allow statistical tools application by LLM. After users have uploaded their dataset, they can ask statistical questions about the dataset, which include but not limited to correlation test, normality test, multicollinearity test and etc.

Next, I integrated a knowledge base as RAG for LLM to learn as part of skills in automate the data modeling process through the application. The RAG contains the data modeling options and recommendations according to the EDA processes conducted by LLM. Once auto mode is enabled, the LLM will decides the most suitable data cleaning approach and data modeling techniques, which include feature selection and hyperparameter tuning. 

Users could also override AI's selection via direct modification, or instruct through the chatbox provided in the left sidebar. If users prefers to conduct data modeling themselves, Auto AI mode could be turned off. Users could then conduct EDA through various summaries and visualization techniques. No worries, the AI assistant is still there to provide hints from time to time. 

