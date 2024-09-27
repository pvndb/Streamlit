import warnings
warnings.filterwarnings('ignore')
import json
import os
import sys
import boto3
import botocore
# from utils import print_ww
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----

# os.environ["AWS_DEFAULT_REGION"] = "us-west-2"  # E.g. "us-east-1"
# os.environ["AWS_PROFILE"] = "demo"
# os.environ["BEDROCK_ASSUME_ROLE"] = "arn:aws:iam::888888:role/corp-dev-use1-foundation-administrator-role"  # E.g. "arn:aws:..."

module_path = ".."
sys.path.append(os.path.abspath(module_path))

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for working with Amazon Bedrock from Python notebooks"""
# Python Built-Ins:
import os
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(read_timeout=1000)
    
    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    
    
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        **client_kwargs
    )
#     endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    
    
    bedrock_agent_client = boto3.client(service_name="bedrock-agent-runtime",
                                        config=retry_config,
                                        endpoint_url="https://bedrock-agent-runtime.us-west-2.amazonaws.com")
    return bedrock_client, bedrock_agent_client

def get_context(retrievalResults):
    context = []
    for retrievedResult in retrievalResults: 
        context.append(retrievedResult['content']['text'])
    return context


 def retrieve(bedrock_agent_client, query, kb_id, numberOfResults=5, metadata_filter=None):
    retrieval_configuration = {
        'vectorSearchConfiguration': {
            'numberOfResults': numberOfResults
        }
    }

    # Add metadata filter if provided
    if metadata_filter:
        retrieval_configuration['vectorSearchConfiguration']['filter'] = metadata_filter

    response = bedrock_agent_client.retrieve(
        retrievalQuery={
            'text': query
        },
        knowledgeBaseId=kb_id,
        retrievalConfiguration=retrieval_configuration
    )
    retrievalResults = response['retrievalResults']
    context = get_context(retrievalResults)
    return response, context   

def model_params(max_tokens=1024, temperature=0):
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0' # change this to use a different version from the model provider
    
    params = {"anthropic_version": "bedrock-2023-05-31",
               "max_tokens": max_tokens,
               "temperature": temperature,
                }
    return modelId, params

def invoke_claude_3(bedrock_client, query, context):
    """
    Invokes Anthropic Claude 3 Sonnet to run an inference using the input
    provided in the request body.

    :param prompt: The prompt that you want Claude 3 to complete.
    :return: Inference response from the model.
    """
    prompt = f"""
        Human: Please use the following context to offer an accurate and concise response to the question below. 
        If you don't know the answer, simply state that you don't know. Avoid adding unreliable information.

        <context>
        {context}
        </context>

        <question>
        {query}
        </question>

        Assistant:"""
    
    modelId, params = model_params()
    messages=[{ "role":'user', "content":[{'type':'text','text': prompt.format(context, query)}]}]
    sonnet_payload = json.dumps({
        "anthropic_version": params["anthropic_version"],
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": 1,
        "messages": messages,
            }  )
    
#     sonnet_payload = json.dumps({
#         "anthropic_version": "bedrock-2023-05-31",
#         "max_tokens": 1024,
#         "messages": messages,
#         "temperature": 0,
#         "top_p": 1
#             }  )
    
    try:
        response = bedrock_client.invoke_model(
            modelId=modelId,
            body=sonnet_payload
        )

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
#         output_list = result.get("content", [])
        result_text = result.get('content')[0]['text']
        print("Invocation details:")
        print(f"- The input length is {input_tokens} tokens.")
        print(f"- The output length is {output_tokens} tokens.")

        return result_text

    except botocore.exceptions.ClientError as err:
        logger.error(
            "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise