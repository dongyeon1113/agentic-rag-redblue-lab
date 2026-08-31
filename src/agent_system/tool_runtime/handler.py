from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from agent_system.contracts import Capability, Principal

RequestT = TypeVar("RequestT", bound=BaseModel)


class ToolHandler(ABC, Generic[RequestT]):
    request_model: type[RequestT]
    capability: Capability

    def validate(self, parameters: dict[str, Any]) -> RequestT:
        return self.request_model.model_validate(parameters)

    @abstractmethod
    async def execute(
        self,
        request: RequestT,
        principal: Principal,
    ) -> dict[str, Any]: ...

