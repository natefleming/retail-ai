import bisect
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from textwrap import dedent
from typing import Annotated, Any, Callable, Optional, Union

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field

from dao_ai.config import AnyVariable, CompositeVariableModel, GenieRoomModel, value_of

MAX_TOKENS_OF_DATA: int = 20000
MAX_ITERATIONS: int = 50
DEFAULT_POLLING_INTERVAL_SECS: int = 2


def _count_tokens(text):
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


@dataclass
class GenieResponse:
    conversation_id: str
    result: Union[str, pd.DataFrame]
    query: Optional[str] = ""
    description: Optional[str] = ""

    def to_json(self):
        return json.dumps(asdict(self))


class GenieToolInput(BaseModel):
    """Input schema for the Genie tool."""

    question: str = Field(
        description="The question to ask Genie about your data. Ask simple, clear questions about your tabular data. For complex analysis, ask multiple simple questions rather than one complex question."
    )


def _truncate_result(dataframe: pd.DataFrame) -> str:
    query_result = dataframe.to_markdown()
    tokens_used = _count_tokens(query_result)

    # If the full result fits, return it
    if tokens_used <= MAX_TOKENS_OF_DATA:
        return query_result.strip()

    def is_too_big(n):
        return _count_tokens(dataframe.iloc[:n].to_markdown()) > MAX_TOKENS_OF_DATA

    # Use bisect_left to find the cutoff point of rows within the max token data limit in a O(log n) complexity
    # Passing True, as this is the target value we are looking for when _is_too_big returns
    cutoff = bisect.bisect_left(range(len(dataframe) + 1), True, key=is_too_big)

    # Slice to the found limit
    truncated_df = dataframe.iloc[:cutoff]

    # Edge case: Cannot return any rows because of tokens so return an empty string
    if len(truncated_df) == 0:
        return ""

    truncated_result = truncated_df.to_markdown()

    # Double-check edge case if we overshot by one
    if _count_tokens(truncated_result) > MAX_TOKENS_OF_DATA:
        truncated_result = truncated_df.iloc[:-1].to_markdown()
    return truncated_result


@mlflow.trace(span_type="PARSER")
def _parse_query_result(resp, truncate_results) -> Union[str, pd.DataFrame]:
    output = resp["result"]
    if not output:
        return "EMPTY"

    columns = resp["manifest"]["schema"]["columns"]
    header = [str(col["name"]) for col in columns]
    rows = []

    for item in output["data_array"]:
        row = []
        for column, value in zip(columns, item):
            type_name = column["type_name"]
            if value is None:
                row.append(None)
                continue

            if type_name in ["INT", "LONG", "SHORT", "BYTE"]:
                row.append(int(value))
            elif type_name in ["FLOAT", "DOUBLE", "DECIMAL"]:
                row.append(float(value))
            elif type_name == "BOOLEAN":
                row.append(value.lower() == "true")
            elif type_name == "DATE" or type_name == "TIMESTAMP":
                row.append(datetime.strptime(value[:10], "%Y-%m-%d").date())
            elif type_name == "BINARY":
                row.append(bytes(value, "utf-8"))
            else:
                row.append(value)

        rows.append(row)

    dataframe = pd.DataFrame(rows, columns=header)

    if truncate_results:
        query_result = _truncate_result(dataframe)
    else:
        query_result = dataframe.to_markdown()

    return query_result.strip()


class Genie:
    def __init__(
        self,
        space_id,
        client: WorkspaceClient | None = None,
        truncate_results: bool = False,
        polling_interval: int = DEFAULT_POLLING_INTERVAL_SECS,
    ):
        self.space_id = space_id
        workspace_client = client or WorkspaceClient()
        self.genie = workspace_client.genie
        self.description = self.genie.get_space(space_id).description
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.truncate_results = truncate_results
        if polling_interval < 1 or polling_interval > 30:
            raise ValueError("poll_interval must be between 1 and 30 seconds")
        self.poll_interval = polling_interval

    @mlflow.trace()
    def start_conversation(self, content):
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/start-conversation",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def create_message(self, conversation_id, content):
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def poll_for_result(self, conversation_id, message_id):
        @mlflow.trace()
        def poll_query_results(attachment_id, query_str, description):
            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self.genie._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result",
                    headers=self.headers,
                )["statement_response"]
                state = resp["status"]["state"]
                if state == "SUCCEEDED":
                    result = _parse_query_result(resp, self.truncate_results)
                    return GenieResponse(
                        conversation_id, result, query_str, description
                    )
                elif state in ["RUNNING", "PENDING"]:
                    logger.debug("Waiting for query result...")
                    time.sleep(self.poll_interval)
                else:
                    return GenieResponse(
                        conversation_id,
                        f"No query result: {resp['state']}",
                        query_str,
                        description,
                    )
            return GenieResponse(
                conversation_id,
                f"Genie query for result timed out after {MAX_ITERATIONS} iterations of {self.poll_interval} seconds",
                query_str,
                description,
            )

        @mlflow.trace()
        def poll_result():
            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self.genie._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}",
                    headers=self.headers,
                )
                if resp["status"] == "COMPLETED":
                    # Check if attachments key exists in response
                    attachments = resp.get("attachments", [])
                    if not attachments:
                        # Handle case where response has no attachments
                        return GenieResponse(
                            conversation_id,
                            result=f"Genie query completed but no attachments found. Response: {resp}",
                        )

                    attachment = next((r for r in attachments if "query" in r), None)
                    if attachment:
                        query_obj = attachment["query"]
                        description = query_obj.get("description", "")
                        query_str = query_obj.get("query", "")
                        attachment_id = attachment["attachment_id"]
                        return poll_query_results(attachment_id, query_str, description)
                    if resp["status"] == "COMPLETED":
                        text_content = next(
                            (r for r in attachments if "text" in r), None
                        )
                        if text_content:
                            return GenieResponse(
                                conversation_id, result=text_content["text"]["content"]
                            )
                        return GenieResponse(
                            conversation_id,
                            result="Genie query completed but no text content found in attachments.",
                        )
                elif resp["status"] in {"CANCELLED", "QUERY_RESULT_EXPIRED"}:
                    return GenieResponse(
                        conversation_id, result=f"Genie query {resp['status'].lower()}."
                    )
                elif resp["status"] == "FAILED":
                    return GenieResponse(
                        conversation_id,
                        result=f"Genie query failed with error: {resp.get('error', 'Unknown error')}",
                    )
                # includes EXECUTING_QUERY, Genie can retry after this status
                else:
                    logger.debug(f"Waiting...: {resp['status']}")
                    time.sleep(self.poll_interval)
            return GenieResponse(
                conversation_id,
                f"Genie query timed out after {MAX_ITERATIONS} iterations of {self.poll_interval} seconds",
            )

        return poll_result()

    @mlflow.trace()
    def ask_question(self, question: str, conversation_id: str | None = None):
        logger.debug(
            f"ask_question called with question: {question}, conversation_id: {conversation_id}"
        )
        if conversation_id:
            resp = self.create_message(conversation_id, question)
        else:
            resp = self.start_conversation(question)
        logger.debug(f"ask_question response: {resp}")
        return self.poll_for_result(resp["conversation_id"], resp["message_id"])


def create_genie_tool(
    genie_room: GenieRoomModel | dict[str, Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    persist_conversation: bool = False,
    truncate_results: bool = False,
    poll_interval: int = DEFAULT_POLLING_INTERVAL_SECS,
) -> Callable[[str], GenieResponse]:
    """
    Create a tool for interacting with Databricks Genie for natural language queries to databases.

    This factory function generates a tool that leverages Databricks Genie to translate natural
    language questions into SQL queries and execute them against retail databases. This enables
    answering questions about inventory, sales, and other structured retail data.

    Args:
        genie_room: GenieRoomModel or dict containing Genie configuration
        name: Optional custom name for the tool. If None, uses default "genie_tool"
        description: Optional custom description for the tool. If None, uses default description

    Returns:
        A LangGraph tool that processes natural language queries through Genie
    """

    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    space_id: AnyVariable = genie_room.space_id or os.environ.get(
        "DATABRICKS_GENIE_SPACE_ID"
    )
    space_id: AnyVariable = genie_room.space_id or os.environ.get(
        "DATABRICKS_GENIE_SPACE_ID"
    )
    if isinstance(space_id, dict):
        space_id = CompositeVariableModel(**space_id)
    space_id = value_of(space_id)

    # genie: Genie = Genie(
    #     space_id=space_id,
    #     client=genie_room.workspace_client,
    #     truncate_results=truncate_results,
    #     polling_interval=poll_interval,
    # )

    default_description: str = dedent("""
    This tool lets you have a conversation and chat with tabular data about <topic>. You should ask
    questions about the data and the tool will try to answer them.
    Please ask simple clear questions that can be answer by sql queries. If you need to do statistics or other forms of testing defer to using another tool.
    Try to ask for aggregations on the data and ask very simple questions.
    Prefer to call this tool multiple times rather than asking a complex question.
    """)

    tool_description: str = (
        description if description is not None else default_description
    )
    tool_name: str = name if name is not None else "genie_tool"

    function_docs = """

Args:
question (str): The question to ask to ask Genie about your data. Ask simple, clear questions about your tabular data. For complex analysis, ask multiple simple questions rather than one complex question.

Returns:
GenieResponse: A response object containing the conversation ID and result from Genie."""
    tool_description = tool_description + function_docs

    @tool(
        name_or_callable=tool_name,
        description=tool_description,
    )
    def genie_tool(
        question: Annotated[str, "The question to ask Genie about your data"],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        genie: Genie = Genie(
            space_id=space_id,
            client=genie_room.workspace_client,
            truncate_results=truncate_results,
            polling_interval=poll_interval,
        )

        """Process a natural language question through Databricks Genie."""
        # Get existing conversation mapping and retrieve conversation ID for this space
        conversation_ids: dict[str, str] = state.get("genie_conversation_ids", {})
        existing_conversation_id: str | None = conversation_ids.get(space_id)
        logger.debug(
            f"Existing conversation ID for space {space_id}: {existing_conversation_id}"
        )

        response: GenieResponse = genie.ask_question(
            question, conversation_id=existing_conversation_id
        )

        current_conversation_id: str = response.conversation_id
        logger.debug(
            f"Current conversation ID for space {space_id}: {current_conversation_id}"
        )

        # Update the conversation mapping with the new conversation ID for this space

        update: dict[str, Any] = {
            "messages": [ToolMessage(response.to_json(), tool_call_id=tool_call_id)],
        }

        if persist_conversation:
            updated_conversation_ids: dict[str, str] = conversation_ids.copy()
            updated_conversation_ids[space_id] = current_conversation_id
            update["genie_conversation_ids"] = updated_conversation_ids

        logger.debug(f"State update: {update}")

        return Command(update=update)

    return genie_tool
