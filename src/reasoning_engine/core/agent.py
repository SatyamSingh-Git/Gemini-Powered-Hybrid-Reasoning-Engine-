# src/reasoning_engine/core/agent.py (THE FINAL WINNING VERSION)

import google.generativeai as genai
from typing import List, Dict, Any
import logging
import json

from ..config import get_settings
from .models import DocumentChunk, Answer, Justification, Evidence
from .tools import AVAILABLE_TOOLS
from .retrievel import HybridRetriever

logger = logging.getLogger(__name__)

try:
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
except Exception as e:
    logger.error(f"Failed to configure Gemini: {e}")


class ReasoningAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.reasoning_model = genai.GenerativeModel(
            'gemini-2.5-flash',
            tools=list(AVAILABLE_TOOLS.values())
        )

    def _execute_tool_call(self, tool_call, chunks) -> List[Dict[str, Any]]:
        tool_name = tool_call.name
        tool_args = dict(tool_call.args)
        if tool_name in AVAILABLE_TOOLS:
            logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
            tool_function = AVAILABLE_TOOLS[tool_name]
            tool_args['clauses'] = chunks
            try:
                return tool_function(**tool_args)
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}': {e}")
                return [{"error": f"Error executing tool: {e}"}]
        logger.warning(f"Tool '{tool_name}' not found.")
        return [{"error": f"Tool '{tool_name}' not found."}]

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting to answer question: '{query}' ---")

        retrieved_chunks = self.retriever.retrieve(query, top_k=7)  # Use a focused set of chunks

        system_prompt = """
        You are a precise, fact-based insurance policy analyst. Your SOLE purpose is to answer user questions based ONLY on the output of the functions you call.

        Your workflow is as follows:
        1.  Analyze the user's question. Call the single most appropriate tool to find the specific data required.
        2.  You will receive the output from the tool as a JSON object.
        3.  Based ONLY on the data within that JSON output, formulate a comprehensive, professional, single-paragraph answer.
        4.  If the tool output is empty or contains an error, you MUST state that the information could not be found in the provided documents.

        **CRITICAL RULES:**
        - DO NOT apologize.
        - DO NOT ask for more information or say "Please provide the document."
        - DO NOT refuse to answer.
        - Synthesize a final answer ONLY from the tool's JSON output.
        """

        chat_session = self.reasoning_model.start_chat()
        initial_message = f"{system_prompt}\n\n**User Question:** {query}"

        response = await chat_session.send_message_async(initial_message)
        response_part = response.candidates[0].content.parts[0]

        final_answer_text = f"The information for '{query}' could not be determined from the policy."

        if response_part.function_call:
            tool_call = response_part.function_call
            tool_results = self._execute_tool_call(tool_call, retrieved_chunks)

            logger.info(f"Tool '{tool_call.name}' returned: {json.dumps(tool_results, indent=2)}")

            if tool_results and "error" not in tool_results[0]:
                # Send the results back to the model for the final synthesis
                synthesis_response = await chat_session.send_message_async(
                    genai.protos.FunctionResponse(
                        name=tool_call.name,
                        response={"result": tool_results}
                    )
                )
                final_answer_text = synthesis_response.text.strip()
        else:
            final_answer_text = response.text.strip()

        logger.info(f"Generated simple answer: {final_answer_text}")

        return Answer(query=query, decision="Information Found", payout=None, justification=[],
                      simple_answer=final_answer_text)