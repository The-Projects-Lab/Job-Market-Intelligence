# Job Market Intelligence Platform

## AI-Powered Job Analytics, Career Intelligence & RAG Assistant

🌐 **Live Application:** http://43.204.45.12:8501/

Streamlit : ** https://job-recommender-ai.streamlit.app/

An end-to-end **Data Engineering + Machine Learning + Generative AI**
platform that automatically collects job postings from multiple
recruitment portals, transforms them into analytics-ready datasets using
a **Medallion Architecture**, applies Machine Learning for career
recommendations, and provides grounded AI answers through
**Retrieval-Augmented Generation (RAG)**.

## Overview

The modern job market generates thousands of new vacancies every day
across multiple recruitment portals. Unfortunately, this information is
scattered, duplicated, inconsistent, and difficult for students or
professionals to interpret.

The **Job Market Intelligence Platform** solves this problem by building
a complete AI-powered career intelligence system.

### The platform automatically:

-   Scrapes job postings from multiple recruitment portals
-   Cleans and standardizes the data using a Medallion Architecture
-   Generates business-ready analytics datasets
-   Trains Machine Learning models for career guidance
-   Uses Retrieval-Augmented Generation (RAG) to answer career questions
    using real job-market data
-   Provides an interactive Streamlit dashboard for visualization and
    AI-powered assistance

Instead of acting like a generic chatbot, the assistant answers
questions using actual job-market data, making responses more reliable,
explainable, and relevant.

## Why This Architecture

Traditional job portals simply display vacancies.

This platform transforms raw job postings into an intelligent data
product by combining modern Data Engineering, Machine Learning, NLP, and
Generative AI into one unified system.

### Key Design Decisions

-   Multi-source web scraping
-   Medallion Architecture (Bronze → Silver → Gold)
-   Three independent ML models
-   Retrieval-Augmented Generation (RAG)
-   ChromaDB vector database
-   Groq API (Llama 3.3 70B Versatile)
-   Prompt Engineering
-   Stateful Conversation Memory
-   Semantic Search using embeddings
-   Streamlit dashboard

## Architecture

``` text
Job Portals
(Shine | Naukri | Foundit | Internshala | RemoteOK)
        │
        ▼
  Web Scrapers
        │
        ▼
 Bronze Layer
        │
        ▼
 Silver Layer
        │
        ▼
 Gold Layer
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Role  Skill   Job
ML    Gap     Recommender
        │
        ▼
 RAG Documents
        │
        ▼
 Text Chunking
        │
        ▼
Embeddings
(all-MiniLM-L6-v2 / Nomic Embed Text v1.5)
        │
        ▼
 ChromaDB
        │
        ▼
 Semantic Search
        │
        ▼
 Prompt Engineering
        │
        ▼
 Stateful Memory
        │
        ▼
Groq API (Llama 3.3 70B Versatile)
        │
        ▼
 Streamlit Dashboard
```

## Features

### Data Engineering

-   Multi-source web scraping
-   ETL pipeline
-   Bronze, Silver & Gold layers
-   Schema standardization
-   Data validation
-   Deduplication

### Market Analytics

-   Skill demand analysis
-   Company hiring trends
-   Salary insights
-   Location analytics
-   Experience distribution

### Machine Learning

1.  Job Role Classification
2.  Skill Gap Analysis
3.  Job Recommendation System (TF-IDF + Cosine Similarity)

### Generative AI

-   RAG
-   ChromaDB
-   Prompt Engineering
-   Stateful Memory
-   Groq API

## NLP Pipeline

-   Text cleaning
-   Skill extraction
-   TF-IDF vectorization
-   Semantic embeddings
-   Chunking
-   Semantic retrieval

## Folder Structure

``` text
job-market-intelligence/
├── config/
├── scrapers/
├── data/
├── etl/
├── models/
├── genai/
├── vector_db/
├── streamlit_app/
├── docs/
├── requirements.txt
└── README.md
```

## Technology Stack

  -----------------------------------------------------------------------
  Category                      Technologies
  ----------------------------- -----------------------------------------
  Programming                   Python

  Data Processing               Pandas, NumPy

  Machine Learning              Scikit-learn, TF-IDF, Linear SVM,
                                Logistic Regression, Naive Bayes

  Generative AI                 Groq API, Llama 3.3 70B Versatile, RAG

  Vector Database               ChromaDB

  Embeddings                    all-MiniLM-L6-v2, Nomic Embed Text v1.5

  UI                            Streamlit, Plotly, Matplotlib
  -----------------------------------------------------------------------

## Future Enhancements

-   Resume Analyzer
-   ATS Resume Scoring
-   Hybrid Search
-   Multi-Agent Career Assistant
-   Cloud Deployment
-   Docker Support

## Business Value

The platform transforms large-scale job postings into actionable career
intelligence by combining Data Engineering, Machine Learning, NLP, and
Generative AI, helping job seekers understand hiring trends, identify
skill gaps, receive personalized recommendations, and interact with
market data through an AI-powered assistant.

## GitHub Highlights

-   80,748+ processed job postings
-   5 recruitment portals
-   Bronze → Silver → Gold Architecture
-   3 Machine Learning models
-   RAG-powered AI assistant
-   ChromaDB vector database
-   Groq API (Llama 3.3 70B Versatile)
-   Prompt Engineering & Stateful Memory
-   Interactive Streamlit Dashboard
