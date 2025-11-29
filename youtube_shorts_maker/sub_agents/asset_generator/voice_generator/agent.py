from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from .tools import generate_narrations

from .prompt import VOICE_GENERATOR_DESCRIPTION, VOICE_GENERATOR_PROMPT

MODEL = LiteLlm(model="openai/gpt-4o")

voice_generator_agent = Agent(
    name="VoiceGeneratorAgent",
    description=VOICE_GENERATOR_DESCRIPTION,
    instruction=VOICE_GENERATOR_PROMPT,
    model=MODEL,
    tools=[generate_narrations],
)
