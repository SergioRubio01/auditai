"""
AVI Month Fixer - Optimized Version
--------------
This module handles the conversion and validation of month formats in AVI Excel files.
Includes parallel processing and caching for optimal performance.
"""

import os
# import logging
import json
import operator
import pandas as pd
import functools
from typing import Annotated, Sequence, List, Literal, Dict, Set
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import base64
from pydantic import BaseModel
from tqdm import tqdm
import asyncio
from datetime import datetime
import aiohttp
from collections import defaultdict


# Load environment variables
load_dotenv()

# Cache for month conversions
MONTH_CACHE = {}

class MES(BaseModel):
    """Model for month validation."""
    MES: Literal['Enero-24', 'Febrero-24', 'Marzo-24', 'Abril-24', 'Mayo-24', 'Junio-24', 
                 'Julio-24', 'Agosto-24', 'Septiembre-24', 'Octubre-24', 'Noviembre-24', 'Diciembre-24']

class AgentState(TypedDict):
    """State for the agents."""
    messages: Annotated[List[BaseMessage], operator.add]
    sender: str

def create_agent(llm, tools, system_message: str):
    """Create an agent with specific tools and system message."""
    base_message = (
        "You are a month format specialist agent. You analyze dates and convert them "
        "to proper Spanish month names. Always return a structured response with the "
        "month name in the MES_CJ field."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            base_message + "\n{system_message}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    
    return prompt | llm.with_structured_output(MES)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    print(result)
    if isinstance(result, ToolMessage):
        pass
    elif isinstance(result, MES):
        result = AIMessage(content=json.dumps({
            "MES_CJ": result.MES,
        }))
    else:
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    
    # Create new state preserving all existing fields
    new_state = state.copy()  # Copy all existing state
    new_state.update({
        "messages": [result],
        "sender": name,
    })
    
    return new_state

def get_month_from_date(date_str: str) -> str:
    """Extract month number from a date string."""
    try:
        # Handle different date formats
        if isinstance(date_str, datetime):
            return str(date_str.month)
        if pd.isna(date_str):
            return None
        
        date_str = str(date_str).strip()
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) >= 2:
                return parts[1] if len(parts[0]) == 4 else parts[0]
        elif '-' in date_str:
            parts = date_str.split('-')
            if len(parts) >= 2:
                return parts[1] if len(parts[0]) == 4 else parts[0]
        # Try to extract any number between 1 and 12
        for part in date_str.split():
            try:
                num = int(part)
                if 1 <= num <= 12:
                    return str(num)
            except ValueError:
                continue
    except:
        pass
    return date_str

MONTH_MAP = {
    '1': 'Enero-24', '01': 'Enero-24',
    '2': 'Febrero-24', '02': 'Febrero-24',
    '3': 'Marzo-24', '03': 'Marzo-24',
    '4': 'Abril-24', '04': 'Abril-24',
    '5': 'Mayo-24', '05': 'Mayo-24',
    '6': 'Junio-24', '06': 'Junio-24',
    '7': 'Julio-24', '07': 'Julio-24',
    '8': 'Agosto-24', '08': 'Agosto-24',
    '9': 'Septiembre-24', '09': 'Septiembre-24',
    '10': 'Octubre-24',
    '11': 'Noviembre-24',
    '12': 'Diciembre-24'
}

async def process_unique_months(unique_months: Set[str], llm) -> Dict[str, str]:
    """Process unique month values to avoid redundant API calls."""
    results = {}
    
    # First try direct mapping
    for month in unique_months:
        month_num = get_month_from_date(month)
        if month_num in MONTH_MAP:
            results[month] = MONTH_MAP[month_num]
            continue
        if month in MONTH_CACHE:
            results[month] = MONTH_CACHE[month]
            continue
    
    # Process remaining months with API
    remaining = [m for m in unique_months if m not in results]
    if not remaining:
        return results
    
    SUB_BATCH_SIZE = 10
    for i in range(0, len(remaining), SUB_BATCH_SIZE):
        sub_batch = remaining[i:i + SUB_BATCH_SIZE]
        try:
            batch_prompt = "\n---\n".join([
                f"Convert this date/month value to Spanish month name: {m}" 
                for m in sub_batch
            ])
            response = await llm.ainvoke([
                HumanMessage(content=f"{batch_prompt}\nRespond with one month name per line, only valid values: Enero-24, Febrero-24, Marzo-24, Abril-24, Mayo-24, Junio-24, Julio-24, Agosto-24, Septiembre-24, Octubre-24, Noviembre-24, Diciembre-24")
            ])
            
            responses = response.content.upper().split('\n')
            for month, resp in zip(sub_batch, responses):
                month_name = None
                for valid_month in MES.__annotations__['MES'].__args__:
                    if valid_month in resp:
                        month_name = valid_month
                        break
                if month_name:
                    results[month] = month_name
                    MONTH_CACHE[month] = month_name
        except Exception as e:
            print(f"Error processing months batch: {str(e)}")
    
    return results

async def process_excel_async(excel_filename: str, sheet_name: str, batch_size: int = 80):
    """Process Excel file with optimized batch processing."""
    # Initialize AI model with optimal settings
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=150
    )
    
    # Read Excel file
    print(f"Reading Excel file: {excel_filename}, sheet: {sheet_name}")
    df = pd.read_excel(excel_filename, sheet_name=sheet_name)
    
    if 'MES_CJ' not in df.columns:
        df['MES_CJ'] = None
    
    # Collect unique month values to process
    unique_months = set()
    for idx, row in df.iterrows():
        if pd.notna(row['Mes']):
            unique_months.add(str(row['Mes']))
        if pd.notna(row['Fecha Inicio']):
            unique_months.add(str(row['Fecha Inicio']))
        if pd.notna(row['Fecha Fin']):
            unique_months.add(str(row['Fecha Fin']))
    
    print(f"Found {len(unique_months)} unique month values to process")
    
    # Process unique months
    month_mapping = await process_unique_months(unique_months, llm)
    
    # Apply results to DataFrame
    print("Applying results to DataFrame...")
    updates = 0
    for idx, row in df.iterrows():
        month = None
        # Try Mes column first
        if pd.notna(row['Mes']) and str(row['Mes']) in month_mapping:
            month = month_mapping[str(row['Mes'])]
        # Try Fecha Inicio if no month found
        elif pd.notna(row['Fecha Inicio']) and str(row['Fecha Inicio']) in month_mapping:
            month = month_mapping[str(row['Fecha Inicio'])]
        # Try Fecha Fin as last resort
        elif pd.notna(row['Fecha Fin']) and str(row['Fecha Fin']) in month_mapping:
            month = month_mapping[str(row['Fecha Fin'])]
        
        if month:
            df.at[idx, 'MES_CJ'] = month
            updates += 1
    
    # Save results
    print(f"Saving results to {excel_filename}")
    with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Processed {len(df)} rows, updated {updates} entries")

async def main():
    """Main entry point for the script."""
    excel_path = "C:/Users/Sergio/Downloads/TOTAL MESES CDI.xlsx"
    sheet_name = "Nominas Desglose"
    batch_size = 80
    
    try:
        await process_excel_async(
            excel_filename=excel_path,
            sheet_name=sheet_name,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 