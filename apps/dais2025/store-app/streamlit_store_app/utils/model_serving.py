"""Model serving utilities for the Streamlit Store App."""

import json
import uuid
from typing import Optional

import streamlit as st
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client


def get_serving_endpoint(endpoint_name: Optional[str] = None) -> str:
    """
    Get the serving endpoint from parameter or config.

    Args:
        endpoint_name: Optional endpoint name to use. If None, gets from config.

    Returns:
        str: The endpoint name to use for model serving

    Raises:
        EnvironmentError: If no endpoint is found in parameter or config
    """
    # Priority: parameter > config
    if endpoint_name:
        return endpoint_name

    # Try to get from config
    config = st.session_state.get("config", {})
    config_endpoint = config.get("model", {}).get("agent_endpoint")
    if config_endpoint:
        return config_endpoint

    raise EnvironmentError(
        "No serving endpoint found. Set endpoint parameter or config.model.agent_endpoint"
    )


def validate_endpoint(endpoint_name: Optional[str] = None) -> str:
    """
    Validate and return the serving endpoint name.

    Args:
        endpoint_name: Optional endpoint name to validate. If None, gets from config/env.

    Returns:
        str: Validated endpoint name

    Raises:
        ValueError: If endpoint name is invalid
    """
    endpoint = get_serving_endpoint(endpoint_name)

    try:
        w = WorkspaceClient()
        w.serving_endpoints.get(endpoint)
        return endpoint
    except Exception as e:
        raise ValueError(f"Invalid serving endpoint: {endpoint}. Error: {str(e)}")


def _throw_unexpected_endpoint_format():
    raise Exception(
        "This app can only run against:"
        "1) Databricks foundation model or external model endpoints with the chat task type (described in https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#chat-completion-model-query)\n"
        "2) Databricks agent serving endpoints that implement the conversational agent schema documented "
        "in https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent"
    )


def query_endpoint_stream(
    messages: list[dict[str, str]],
    max_tokens: int,
    return_traces: bool,
    endpoint_name: Optional[str] = None,
    custom_inputs: Optional[dict] = None,
):
    """
    Streams chat-completions style chunks and converts to ChatAgent-style streaming deltas.

    Args:
        messages: List of message dictionaries with role and content
        max_tokens: Maximum tokens in response
        return_traces: Whether to return tracing information
        endpoint_name: Optional endpoint name. If None, gets from config/env.
        custom_inputs: Optional custom inputs to pass to the agent (e.g., store_num, user_id)
    """
    endpoint = validate_endpoint(endpoint_name)
    client = get_deploy_client("databricks")

    # Prepare input payload
    inputs = {
        "messages": messages,
        "max_tokens": max_tokens,
    }

    # Add custom_inputs if provided
    if custom_inputs:
        inputs["custom_inputs"] = custom_inputs

    if return_traces:
        inputs["databricks_options"] = {"return_trace": True}

    stream_id = str(uuid.uuid4())  # Generate unique ID for the stream

    for chunk in client.predict_stream(endpoint=endpoint, inputs=inputs):
        if "choices" in chunk:
            choices = chunk["choices"]
            if len(choices) > 0:
                # Convert from chat completions to ChatAgent format
                content = choices[0]["delta"].get("content", "")
                if content:
                    yield {
                        "delta": {
                            "role": "assistant",
                            "content": content,
                            "id": stream_id,
                        },
                    }
        elif "delta" in chunk:
            # Yield the ChatAgentChunk directly
            yield chunk
        else:
            _throw_unexpected_endpoint_format()


def query_endpoint(
    messages: list[dict[str, str]],
    endpoint_name: Optional[str] = None,
    custom_inputs: Optional[dict] = None,
    **kwargs,
):
    """
    Query an endpoint, returning the string message content and request ID for feedback.

    Args:
        messages (list): List of message dictionaries with role and content
        endpoint_name (str, optional): Name of the model serving endpoint. If None, gets from config/env.
        custom_inputs (dict, optional): Custom inputs to pass to the agent (e.g., store_num, user_id)
        **kwargs: Optional parameters including:
            - return_traces (bool): Whether to return tracing information
            - max_tokens (int): Maximum tokens in response
            - temperature (float): Sampling temperature
            - stop (list): List of stop sequences
            - n (int): Number of completions
            - stream (bool): Whether to stream the response
    """
    endpoint = validate_endpoint(endpoint_name)

    # Start with required messages parameter
    inputs = {"messages": messages}

    # Add custom_inputs if provided
    if custom_inputs:
        inputs["custom_inputs"] = custom_inputs

    # Add optional parameters from kwargs
    for param in ["temperature", "max_tokens", "stop", "n", "stream"]:
        if param in kwargs:
            inputs[param] = kwargs[param]

    # Handle return_traces separately as it's not a model parameter
    if kwargs.get("return_traces"):
        inputs["databricks_options"] = {"return_trace": True}

    res = get_deploy_client("databricks").predict(
        endpoint=endpoint,
        inputs=inputs,
    )

    request_id = res.get("databricks_output", {}).get("databricks_request_id")
    if "messages" in res:
        return res["messages"], request_id
    elif "choices" in res:
        return [res["choices"][0]["message"]], request_id
    _throw_unexpected_endpoint_format()


def submit_feedback(
    request_id: str, rating: Optional[int] = None, endpoint_name: Optional[str] = None
):
    """
    Submit feedback to the agent.

    Args:
        request_id: ID of the request to provide feedback for
        rating: Optional rating (1 for positive, 0 for negative)
        endpoint_name: Optional endpoint name. If None, gets from config/env.
    """
    endpoint = validate_endpoint(endpoint_name)

    rating_string = "positive" if rating == 1 else "negative" if rating == 0 else None
    text_assessments = (
        []
        if rating is None
        else [
            {
                "ratings": {
                    "answer_correct": {"value": rating_string},
                },
                "free_text_comment": None,
            }
        ]
    )

    proxy_payload = {
        "dataframe_records": [
            {
                "source": json.dumps({"id": "e2e-chatbot-app", "type": "human"}),
                "request_id": request_id,
                "text_assessments": json.dumps(text_assessments),
                "retrieval_assessments": json.dumps([]),
            }
        ]
    }
    w = WorkspaceClient()
    return w.api_client.do(
        method="POST",
        path=f"/serving-endpoints/{endpoint}/served-models/feedback/invocations",
        body=proxy_payload,
    )


def endpoint_supports_feedback(endpoint_name: Optional[str] = None) -> bool:
    """
    Check if an endpoint supports feedback.

    Args:
        endpoint_name: Optional endpoint name. If None, gets from config/env.

    Returns:
        bool: Whether the endpoint supports feedback
    """
    try:
        endpoint = validate_endpoint(endpoint_name)
        w = WorkspaceClient()
        endpoint_info = w.serving_endpoints.get(endpoint)
        return "feedback" in [
            entity.entity_name for entity in endpoint_info.config.served_entities
        ]
    except:
        return False
