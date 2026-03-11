import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from data.raw import prompts_recipes


class PromptEngineeringService:
    def __init__(
        self,
        model: str = os.environ.get("MODEL_NAME", "claude-3-5-sonnet-20241022"),
        temperature: float = 0.2,
        system_prompt: str = prompts_recipes.SYSTEM_INSTRUCTIONS,
    ):
        self.model: str = model
        self.temperature: float = temperature
        self.system_prompt: str = system_prompt
        self.llm: BaseChatModel = self._initialize_llm()

    def _initialize_llm(self) -> BaseChatModel | None:
        """Return a chat model for known name prefixes, else ``None``.

        Uses ``ChatOpenAI`` for ``gpt-``/``o1-`` models and ``ChatAnthropic``
        for ``claude-`` models. Provider SDKs read API keys from environment.
        """
        if self.model.startswith("gpt-") or self.model.startswith("o1-"):
            # ChatOpenAI automatically reads OPENAI_API_KEY from environment
            return ChatOpenAI(model=self.model, temperature=self.temperature)
        elif self.model.startswith("claude-"):
            # ChatAnthropic automatically reads ANTHROPIC_API_KEY from environment
            return ChatAnthropic(model_name=self.model, temperature=self.temperature)
        return None

    def answer_anfrage(self, anfrage: str) -> str:
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(self.system_prompt),
                    HumanMessagePromptTemplate.from_template("{request}"),
                ]
            )
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"request": anfrage})
