"""
LLM Council - Multi-Model Fusion System for Claude Codex.

Inspired by Karpathy's llm-council, this module implements a 3-stage consensus
mechanism that queries multiple LLMs, has them review each other's responses
anonymously, and synthesizes a superior final answer.

The 3 Stages:
1. RESPOND: Query all council members in parallel, collect individual responses
2. REVIEW: Each model reviews and ranks other responses (anonymized)
3. SYNTHESIZE: Chairman model fuses insights into a final superior response

Usage:
    Set LLM_BACKEND=council and configure COUNCIL_MODELS, COUNCIL_CHAIRMAN

Example config:
    COUNCIL_MODELS=gemini-2.0-flash,gpt-4o,claude-sonnet-4
    COUNCIL_CHAIRMAN=claude-sonnet-4
    COUNCIL_MODE=full  # or "fast" to skip review stage
"""

import os
import re
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from claudecodex.models import (
    MessagesRequest,
    MessagesResponse,
    Message,
    TokenCountRequest,
    TokenCountResponse,
)

logger = logging.getLogger(__name__)


# === CONFIGURATION ===


@dataclass
class CouncilMember:
    """Represents a single LLM in the council."""

    name: str  # Display name (e.g., "gemini", "gpt4", "claude")
    backend: str  # Backend type: "openai_compatible" or "bedrock"
    model_id: str  # Actual model identifier
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    weight: float = 1.0  # Voting weight (for future weighted consensus)


@dataclass
class CouncilConfig:
    """Configuration for the LLM Council."""

    members: List[CouncilMember] = field(default_factory=list)
    chairman: Optional[CouncilMember] = None
    mode: str = (
        "full"  # "full" = all 3 stages, "fast" = skip review, "race" = first wins
    )
    timeout: int = 120  # seconds per model
    max_parallel: int = 5  # max concurrent requests


def get_council_config() -> CouncilConfig:
    """Build council configuration from environment variables."""

    # Parse COUNCIL_MODELS: comma-separated list of model specs
    # Format: "provider:model" or just "model" (provider inferred)
    # Examples:
    #   COUNCIL_MODELS=gemini-2.0-flash,gpt-4o,claude-sonnet-4
    #   COUNCIL_MODELS=openai:gpt-4o,bedrock:claude-sonnet-4,gemini:gemini-2.0-flash

    models_str = os.environ.get("COUNCIL_MODELS", "")
    if not models_str:
        raise ValueError(
            "COUNCIL_MODELS environment variable required for council mode"
        )

    members = []
    for spec in models_str.split(","):
        spec = spec.strip()
        if not spec:
            continue

        member = _parse_model_spec(spec)
        if member:
            members.append(member)

    if len(members) < 2:
        raise ValueError("Council requires at least 2 models in COUNCIL_MODELS")

    # Parse chairman (defaults to first member)
    chairman_str = os.environ.get("COUNCIL_CHAIRMAN", "")
    if chairman_str:
        chairman = _parse_model_spec(chairman_str)
    else:
        chairman = members[0]  # Default: first model is chairman

    # Parse mode
    mode = os.environ.get("COUNCIL_MODE", "full").lower()
    if mode not in ["full", "fast", "race"]:
        logger.warning(f"Unknown COUNCIL_MODE '{mode}', defaulting to 'full'")
        mode = "full"

    return CouncilConfig(
        members=members,
        chairman=chairman,
        mode=mode,
        timeout=int(os.environ.get("COUNCIL_TIMEOUT", "120")),
        max_parallel=int(os.environ.get("COUNCIL_MAX_PARALLEL", "5")),
    )


def _parse_model_spec(spec: str) -> Optional[CouncilMember]:
    """Parse a model specification string into a CouncilMember."""

    # Handle "provider:model" format
    if ":" in spec:
        provider, model = spec.split(":", 1)
        provider = provider.lower().strip()
        model = model.strip()
    else:
        model = spec.strip()
        provider = _infer_provider(model)

    # Map provider to backend and configure
    if provider in ["openai", "gpt"]:
        return CouncilMember(
            name=f"gpt_{model.replace('-', '_')}",
            backend="openai_compatible",
            model_id=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )
    elif provider in ["gemini", "google"]:
        return CouncilMember(
            name=f"gemini_{model.replace('-', '_')}",
            backend="openai_compatible",
            model_id=model,
            api_key=os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAICOMPATIBLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
    elif provider in ["anthropic", "claude", "bedrock"]:
        return CouncilMember(
            name=f"claude_{model.replace('-', '_')}",
            backend="bedrock",
            model_id=_get_bedrock_model_id(model),
        )
    elif provider in ["ollama", "local"]:
        return CouncilMember(
            name=f"local_{model.replace('-', '_')}",
            backend="openai_compatible",
            model_id=model,
            api_key="ollama",  # Ollama doesn't need real key
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    else:
        # Generic OpenAI-compatible
        return CouncilMember(
            name=f"{provider}_{model.replace('-', '_')}",
            backend="openai_compatible",
            model_id=model,
            api_key=os.environ.get(f"{provider.upper()}_API_KEY")
            or os.environ.get("OPENAICOMPATIBLE_API_KEY"),
            base_url=os.environ.get(f"{provider.upper()}_BASE_URL"),
        )


def _infer_provider(model: str) -> str:
    """Infer provider from model name."""
    model_lower = model.lower()

    if "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    elif "gemini" in model_lower:
        return "gemini"
    elif (
        "claude" in model_lower
        or "sonnet" in model_lower
        or "opus" in model_lower
        or "haiku" in model_lower
    ):
        return "bedrock"
    elif "llama" in model_lower or "mistral" in model_lower or "qwen" in model_lower:
        return "ollama"
    else:
        return "openai_compatible"


def _get_bedrock_model_id(model: str) -> str:
    """Convert short model name to full Bedrock model ID."""
    model_lower = model.lower()

    # Map common short names to full Bedrock IDs
    bedrock_models = {
        "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-haiku-4": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
        "haiku-4": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    }

    return bedrock_models.get(model_lower, model)


# === STAGE 1: PARALLEL RESPONSE COLLECTION ===


@dataclass
class MemberResponse:
    """Response from a single council member."""

    member: CouncilMember
    response: Optional[MessagesResponse] = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.response is not None and self.error is None

    @property
    def text(self) -> str:
        """Extract text content from response."""
        if not self.response:
            return ""
        texts = []
        for block in self.response.content:
            if hasattr(block, "text"):
                texts.append(block.text)
        return "\n".join(texts)


def _call_member(member: CouncilMember, request: MessagesRequest) -> MemberResponse:
    """Call a single council member and return its response."""
    start_time = time.time()

    try:
        if member.backend == "bedrock":
            response = _call_bedrock_member(member, request)
        else:
            response = _call_openai_member(member, request)

        latency_ms = (time.time() - start_time) * 1000
        return MemberResponse(member=member, response=response, latency_ms=latency_ms)

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Council member {member.name} failed: {str(e)}")
        return MemberResponse(member=member, error=str(e), latency_ms=latency_ms)


def _call_bedrock_member(
    member: CouncilMember, request: MessagesRequest
) -> MessagesResponse:
    """Call a Bedrock-based council member."""
    import boto3
    from botocore.config import Config
    from claudecodex.bedrock import (
        convert_to_bedrock_messages,
        extract_system_message,
        create_claude_response,
    )

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    config = Config(region_name=region, retries={"max_attempts": 2, "mode": "adaptive"})
    profile_name = os.environ.get("AWS_PROFILE", "saml")
    session = boto3.Session(profile_name=profile_name)
    client = session.client("bedrock-runtime", region_name=region, config=config)

    bedrock_messages = convert_to_bedrock_messages(request)
    system_message = extract_system_message(request)

    converse_params = {
        "modelId": member.model_id,
        "messages": bedrock_messages,
        "inferenceConfig": {
            "temperature": request.temperature or 0.7,
            "maxTokens": request.max_tokens,
        },
    }

    if system_message:
        converse_params["system"] = [{"text": system_message}]

    # Handle tools if present
    if request.tools:
        tool_config = {"tools": []}
        for tool in request.tools:
            tool_config["tools"].append(
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": {"json": tool.input_schema},
                    }
                }
            )
        converse_params["toolConfig"] = tool_config

    response = client.converse(**converse_params)
    return create_claude_response(response, member.model_id)


def _call_openai_member(
    member: CouncilMember, request: MessagesRequest
) -> MessagesResponse:
    """Call an OpenAI-compatible council member."""
    import requests
    from claudecodex.openai_compatible import (
        convert_to_openai_messages,
        convert_tools_to_openai,
        create_claude_response_from_openai,
    )

    if not member.api_key:
        raise ValueError(f"No API key for member {member.name}")
    if not member.base_url:
        raise ValueError(f"No base URL for member {member.name}")

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {member.api_key}",
            "Content-Type": "application/json",
        }
    )

    openai_messages = convert_to_openai_messages(request)

    payload = {
        "model": member.model_id,
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature or 0.7,
    }

    if request.tools:
        payload["tools"] = convert_tools_to_openai(request.tools)

    response = session.post(
        f"{member.base_url}/chat/completions", json=payload, timeout=120
    )
    response.raise_for_status()

    return create_claude_response_from_openai(response.json(), member.model_id)


def collect_responses(
    config: CouncilConfig, request: MessagesRequest
) -> List[MemberResponse]:
    """Stage 1: Query all council members in parallel."""
    logger.info(
        f"[Council Stage 1] Querying {len(config.members)} models in parallel..."
    )

    responses = []

    with ThreadPoolExecutor(max_workers=config.max_parallel) as executor:
        futures = {
            executor.submit(_call_member, member, request): member
            for member in config.members
        }

        for future in as_completed(futures, timeout=config.timeout):
            try:
                result = future.result()
                responses.append(result)

                if result.success:
                    logger.info(
                        f"  [{result.member.name}] responded in {result.latency_ms:.0f}ms"
                    )
                else:
                    logger.warning(f"  [{result.member.name}] failed: {result.error}")

            except Exception as e:
                member = futures[future]
                logger.error(f"  [{member.name}] exception: {str(e)}")
                responses.append(MemberResponse(member=member, error=str(e)))

    successful = sum(1 for r in responses if r.success)
    logger.info(
        f"[Council Stage 1] Complete: {successful}/{len(config.members)} succeeded"
    )

    return responses


# === STAGE 2: ANONYMOUS PEER REVIEW ===

REVIEW_PROMPT_TEMPLATE = """You are evaluating responses from multiple AI assistants to the same question.

ORIGINAL QUESTION:
{question}

RESPONSES TO EVALUATE:
{responses}

INSTRUCTIONS:
1. Analyze each response for: accuracy, completeness, clarity, and helpfulness
2. Consider code quality if code is present (correctness, best practices, security)
3. Identify unique insights or approaches in each response
4. Note any errors, hallucinations, or missing information

Provide your evaluation in this exact format:

EVALUATION:
- Response A: [Brief analysis]
- Response B: [Brief analysis]
- Response C: [Brief analysis]
...

FINAL RANKING:
1. [Letter] - [One sentence reason]
2. [Letter] - [One sentence reason]
3. [Letter] - [One sentence reason]
...

BEST INSIGHTS:
- [Key insight from any response that should be preserved]
- [Another key insight]
..."""


def _anonymize_responses(responses: List[MemberResponse]) -> Tuple[str, Dict[str, str]]:
    """Convert responses to anonymous format for peer review."""
    successful = [r for r in responses if r.success]

    label_to_member = {}
    formatted_responses = []

    for i, resp in enumerate(successful):
        label = chr(65 + i)  # A, B, C, ...
        label_to_member[label] = resp.member.name

        formatted_responses.append(f"=== Response {label} ===\n{resp.text}\n")

    return "\n".join(formatted_responses), label_to_member


def _parse_rankings(
    review_text: str, label_to_member: Dict[str, str]
) -> List[Tuple[str, int]]:
    """Parse rankings from review response."""
    rankings = []

    # Look for "FINAL RANKING:" section
    ranking_match = re.search(
        r"FINAL RANKING:(.*?)(?:BEST INSIGHTS:|$)",
        review_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not ranking_match:
        return rankings

    ranking_section = ranking_match.group(1)

    # Parse each ranking line: "1. A - reason" or "1. A"
    for line in ranking_section.strip().split("\n"):
        match = re.match(r"(\d+)\.\s*([A-Z])", line.strip())
        if match:
            rank = int(match.group(1))
            label = match.group(2)
            if label in label_to_member:
                rankings.append((label_to_member[label], rank))

    return rankings


def _parse_insights(review_text: str) -> List[str]:
    """Extract key insights from review response."""
    insights = []

    # Look for "BEST INSIGHTS:" section
    insights_match = re.search(
        r"BEST INSIGHTS:(.*?)$", review_text, re.DOTALL | re.IGNORECASE
    )
    if not insights_match:
        return insights

    insights_section = insights_match.group(1)

    # Parse bullet points
    for line in insights_section.strip().split("\n"):
        line = line.strip()
        if line.startswith("-") or line.startswith("*"):
            insight = line[1:].strip()
            if insight:
                insights.append(insight)

    return insights


@dataclass
class ReviewResult:
    """Result of peer review stage."""

    reviewer: str
    rankings: List[Tuple[str, int]]  # (member_name, rank)
    insights: List[str]
    raw_review: str


def conduct_reviews(
    config: CouncilConfig, request: MessagesRequest, responses: List[MemberResponse]
) -> List[ReviewResult]:
    """Stage 2: Have each model review others' responses anonymously."""

    successful = [r for r in responses if r.success]
    if len(successful) < 2:
        logger.warning(
            "[Council Stage 2] Skipping review - need at least 2 successful responses"
        )
        return []

    logger.info(
        f"[Council Stage 2] Conducting peer reviews with {len(successful)} reviewers..."
    )

    # Extract original question from request
    question = ""
    for msg in request.messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                question = msg.content
            elif hasattr(msg.content, "__iter__"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        question = block.text
                        break
            break

    # Anonymize responses
    anonymous_responses, label_to_member = _anonymize_responses(responses)

    # Build review prompt
    review_prompt = REVIEW_PROMPT_TEMPLATE.format(
        question=question, responses=anonymous_responses
    )

    # Create review request (no tools, just analysis)
    review_request = MessagesRequest(
        model=request.model,
        max_tokens=2000,
        messages=[Message(role="user", content=review_prompt)],
        temperature=0.3,  # Lower temp for more consistent reviews
    )

    results = []

    # Have each successful member conduct a review
    with ThreadPoolExecutor(max_workers=config.max_parallel) as executor:
        futures = {}
        for resp in successful:
            future = executor.submit(_call_member, resp.member, review_request)
            futures[future] = resp.member

        for future in as_completed(futures, timeout=config.timeout):
            member = futures[future]
            try:
                review_response = future.result()
                if review_response.success:
                    review_text = review_response.text
                    rankings = _parse_rankings(review_text, label_to_member)
                    insights = _parse_insights(review_text)

                    results.append(
                        ReviewResult(
                            reviewer=member.name,
                            rankings=rankings,
                            insights=insights,
                            raw_review=review_text,
                        )
                    )
                    logger.info(f"  [{member.name}] completed review")
                else:
                    logger.warning(
                        f"  [{member.name}] review failed: {review_response.error}"
                    )

            except Exception as e:
                logger.error(f"  [{member.name}] review exception: {str(e)}")

    logger.info(f"[Council Stage 2] Complete: {len(results)} reviews collected")
    return results


def aggregate_rankings(reviews: List[ReviewResult]) -> Dict[str, float]:
    """Aggregate rankings from multiple reviewers into scores."""
    scores = {}
    counts = {}

    for review in reviews:
        for member_name, rank in review.rankings:
            if member_name not in scores:
                scores[member_name] = 0.0
                counts[member_name] = 0

            # Convert rank to score (lower rank = higher score)
            # Using inverse ranking: 1st = 1.0, 2nd = 0.5, 3rd = 0.33, etc.
            score = 1.0 / rank
            scores[member_name] += score
            counts[member_name] += 1

    # Average the scores
    for member in scores:
        if counts[member] > 0:
            scores[member] /= counts[member]

    return scores


def collect_all_insights(reviews: List[ReviewResult]) -> List[str]:
    """Collect and deduplicate insights from all reviews."""
    all_insights = []
    seen = set()

    for review in reviews:
        for insight in review.insights:
            # Simple deduplication by lowercase first 50 chars
            key = insight.lower()[:50]
            if key not in seen:
                seen.add(key)
                all_insights.append(insight)

    return all_insights


# === STAGE 3: CHAIRMAN SYNTHESIS ===

SYNTHESIS_PROMPT_TEMPLATE = """You are the Chairman of an AI council. Multiple AI models have responded to a user's question, and peer reviews have been conducted. Your task is to synthesize the BEST possible response.

ORIGINAL QUESTION:
{question}

INDIVIDUAL RESPONSES:
{responses}

AGGREGATE RANKINGS (higher score = better):
{rankings}

KEY INSIGHTS IDENTIFIED:
{insights}

INSTRUCTIONS:
1. Synthesize the best elements from all responses
2. Prioritize accuracy and correctness - verify any claims
3. Include the most valuable insights identified
4. If responses disagree, favor the higher-ranked response's approach
5. Maintain the style appropriate for the question (code, explanation, etc.)
6. Do NOT mention that you are synthesizing from multiple sources
7. Respond directly as if you are answering the original question

YOUR SYNTHESIZED RESPONSE:"""


def synthesize_response(
    config: CouncilConfig,
    request: MessagesRequest,
    responses: List[MemberResponse],
    reviews: List[ReviewResult],
) -> MessagesResponse:
    """Stage 3: Chairman synthesizes final response from all inputs."""

    successful = [r for r in responses if r.success]
    if not successful:
        raise ValueError("No successful responses to synthesize")

    # If only one response or in fast/race mode, return best response directly
    if len(successful) == 1 or config.mode == "race":
        logger.info(
            "[Council Stage 3] Single/race mode - returning best response directly"
        )
        return successful[0].response

    logger.info(
        f"[Council Stage 3] Chairman ({config.chairman.name}) synthesizing final response..."
    )

    # Extract original question
    question = ""
    for msg in request.messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                question = msg.content
            elif hasattr(msg.content, "__iter__"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        question = block.text
                        break
            break

    # Format responses with member names
    formatted_responses = []
    for resp in successful:
        formatted_responses.append(
            f"=== {resp.member.name} ({resp.latency_ms:.0f}ms) ===\n{resp.text}\n"
        )

    # Aggregate rankings if we have reviews
    rankings_str = "No peer reviews conducted"
    insights_str = "No insights collected"

    if reviews:
        scores = aggregate_rankings(reviews)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings_str = "\n".join(
            [f"  {name}: {score:.2f}" for name, score in sorted_scores]
        )

        all_insights = collect_all_insights(reviews)
        if all_insights:
            insights_str = "\n".join(
                [f"  - {insight}" for insight in all_insights[:10]]
            )

    # Build synthesis prompt
    synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        question=question,
        responses="\n".join(formatted_responses),
        rankings=rankings_str,
        insights=insights_str,
    )

    # Create synthesis request
    synthesis_request = MessagesRequest(
        model=request.model,
        max_tokens=request.max_tokens,
        messages=[Message(role="user", content=synthesis_prompt)],
        system=request.system,
        temperature=request.temperature or 0.7,
        tools=request.tools,  # Pass through tools in case synthesis needs them
        tool_choice=request.tool_choice,
    )

    # Call chairman for final synthesis
    chairman_response = _call_member(config.chairman, synthesis_request)

    if chairman_response.success:
        logger.info(
            f"[Council Stage 3] Synthesis complete in {chairman_response.latency_ms:.0f}ms"
        )

        # Add metadata about the council process to usage
        response = chairman_response.response

        # Calculate total tokens across all responses
        total_input = sum(
            r.response.usage.input_tokens for r in successful if r.response
        )
        total_output = sum(
            r.response.usage.output_tokens for r in successful if r.response
        )

        # Update usage to reflect total council usage
        response.usage.input_tokens = total_input + response.usage.input_tokens
        response.usage.output_tokens = total_output + response.usage.output_tokens

        return response
    else:
        # Fallback: return highest-ranked response if synthesis fails
        logger.warning(f"[Council Stage 3] Synthesis failed: {chairman_response.error}")
        logger.info(
            "[Council Stage 3] Falling back to highest-ranked individual response"
        )

        if reviews:
            scores = aggregate_rankings(reviews)
            best_member = max(scores, key=scores.get) if scores else None
            for resp in successful:
                if resp.member.name == best_member:
                    return resp.response

        # Ultimate fallback: first successful response
        return successful[0].response


# === MAIN COUNCIL ENTRY POINT ===


def call_council(request: MessagesRequest) -> MessagesResponse:
    """
    Execute a request through the LLM Council.

    This is the main entry point that orchestrates all 3 stages:
    1. Collect responses from all council members in parallel
    2. Conduct anonymous peer reviews (if mode=full)
    3. Chairman synthesizes final response

    Returns a MessagesResponse that appears like a normal Claude response.
    """
    start_time = time.time()

    # Load configuration
    config = get_council_config()

    logger.info(
        f"[Council] Starting with {len(config.members)} members, mode={config.mode}"
    )
    logger.info(f"[Council] Members: {', '.join(m.name for m in config.members)}")
    logger.info(f"[Council] Chairman: {config.chairman.name}")

    # Stage 1: Collect responses
    responses = collect_responses(config, request)

    successful = [r for r in responses if r.success]
    if not successful:
        raise ValueError("All council members failed to respond")

    # Race mode: return first successful response
    if config.mode == "race":
        fastest = min(successful, key=lambda r: r.latency_ms)
        logger.info(
            f"[Council] Race mode: {fastest.member.name} won in {fastest.latency_ms:.0f}ms"
        )
        return fastest.response

    # Stage 2: Peer reviews (skip in fast mode)
    reviews = []
    if config.mode == "full" and len(successful) >= 2:
        reviews = conduct_reviews(config, request, responses)

    # Stage 3: Synthesize final response
    final_response = synthesize_response(config, request, responses, reviews)

    total_time = (time.time() - start_time) * 1000
    logger.info(f"[Council] Complete in {total_time:.0f}ms total")

    return final_response


def count_council_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Estimate token count for council requests."""
    # Use simple estimation - actual count depends on all models used
    from claudecodex.bedrock import count_tokens_from_messages

    base_count = count_tokens_from_messages(request.messages, request.system)

    # Council uses more tokens due to review and synthesis stages
    # Rough estimate: 3x base for full mode, 1.5x for fast mode
    config = get_council_config()
    multiplier = 3.0 if config.mode == "full" else 1.5

    return TokenCountResponse(input_tokens=int(base_count * multiplier))


# === COUNCIL INFO ===


def get_council_info() -> dict:
    """Get information about the current council configuration."""
    try:
        config = get_council_config()
        return {
            "backend": "council",
            "mode": config.mode,
            "members": [
                {"name": m.name, "model": m.model_id, "backend": m.backend}
                for m in config.members
            ],
            "chairman": {
                "name": config.chairman.name,
                "model": config.chairman.model_id,
            },
            "timeout": config.timeout,
        }
    except Exception as e:
        return {"backend": "council", "error": str(e)}
