# src/reasoning_engine/core/agent.py (THE SUPER-ANALYST AGENT)

import google.generativeai as genai
from typing import List, Dict, Any
import logging
import json
import asyncio
import collections.abc

from ..config import get_settings
from .models import DocumentChunk, Answer
from .tools import AVAILABLE_TOOLS
from .retrievel import HybridRetriever

logger = logging.getLogger(__name__)

try:
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)
except Exception as e:
    logger.error(f"Failed to configure Gemini: {e}")

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, collections.abc.Mapping)):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(vars(obj))
    else:
        return obj

class ReasoningAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            tools=list(AVAILABLE_TOOLS.values())
        )

    def _execute_tool_call(self, tool_call, chunks) -> List[Dict[str, Any]]:
        tool_name = tool_call.name
        tool_args = dict(tool_call.args)
        if tool_name in AVAILABLE_TOOLS:
            logger.info(f"EXECUTOR: Calling tool '{tool_name}' with args: {tool_args}")
            tool_function = AVAILABLE_TOOLS[tool_name]
            tool_args['clauses'] = chunks
            try:
                return tool_function(**tool_args)
            except Exception as e:
                return [{"error": f"Error executing tool: {e}"}]
        return [{"error": f"Tool '{tool_name}' not found."}]

    async def answer_question(self, query: str) -> Answer:
        logger.info(f"--- Starting Super-Analyst process for question: '{query}' ---")

        # PHASE 1: THE PLANNER
        planning_prompt = f"""
        You are a meticulous research planner. Your task is to create a step-by-step plan to answer the user's question about an insurance policy.
        Each step in your plan MUST be a call to one of the available tools.
        Think step-by-step to break down the user's query into the necessary tool calls.
        For example, for "What is the waiting period for pre-existing diseases?", a good plan is:
        1. Call `find_information` with topic="pre-existing diseases" to get the precise definition and context.
        2. Call `check_waiting_period` with procedure_terms=["pre-existing diseases"] to find the specific duration mentioned in that context.
        For a simple question like "What is the grace period?", a single step is enough:
        1. Call `check_waiting_period` with procedure_terms=["grace period", "premium payment"].

        User Question: "{query}"
        Create the research plan now.
        """

        chat_session = self.model.start_chat()
        plan_response = await chat_session.send_message_async(planning_prompt)

        # PHASE 2: THE EXECUTOR
        gathered_evidence = []
        try:
            plan_tool_calls = plan_response.candidates[0].content.parts
            logger.info(f"PLANNER: Generated a plan with {len(plan_tool_calls)} steps.")

            for part in plan_tool_calls:
                if not part.function_call:
                    continue

                tool_call = part.function_call
                search_query = " ".join(str(v) for v in dict(tool_call.args).values())
                retrieved_chunks = self.retriever.retrieve(search_query, top_k=7)

                tool_results = self._execute_tool_call(tool_call, retrieved_chunks)

                if tool_results and "error" not in tool_results[0]:
                    gathered_evidence.append({
                        "step_executed": tool_call.name,
                        "parameters": {key: value for key, value in tool_call.args.items()},
                        "evidence_found": tool_results
                    })
        except (ValueError, IndexError) as e:
            logger.error(f"PLANNER: Could not parse a valid plan from the model's response. Error: {e}")
            gathered_evidence.append({"error": "Failed to create a valid research plan."})

        # PHASE 3: THE SYNTHESIZER
        serializable_evidence = make_json_serializable(gathered_evidence)
        synthesis_prompt = f"""
        You are an expert insurance analyst. Your final and only job is to synthesize the following collected research evidence into a single, comprehensive, and professional paragraph.
        The user's original question was: "{query}"

        Here is the evidence dossier you have gathered:
        ---
        {json.dumps(serializable_evidence, indent=2)}
        ---

        Based ONLY on the evidence in the dossier, write the definitive final answer.
        - If the evidence contains the answer, state it directly and confidently.
        - **If the evidence seems incomplete, synthesize what you can and note what is missing. For example, if you find the definition of AYUSH but not the coverage limit, state the definition and then say the specific coverage limit could not be determined.**
        - If the evidence is empty or contains errors, you MUST state that the information could not be found in the provided policy documents.
        - Do not apologize, do not ask for more information, and do not refuse to answer.
        """

        final_response = await self.model.generate_content_async(synthesis_prompt)
        final_answer_text = final_response.text.strip()

        logger.info(f"SYNTHESIZER: Generated final answer: {final_answer_text}")
        return Answer(query=query, decision="Information Found", payout=None, justification=[],
                      simple_answer=final_answer_text)

    async def answer_all_questions(self, questions: List[str]) -> List[Answer]:
        answers = []
        for query in questions:
            answer = await self.answer_question(query)
            answers.append(answer)
            await asyncio.sleep(1)
        return answers