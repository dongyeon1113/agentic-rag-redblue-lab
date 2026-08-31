from __future__ import annotations

import httpx

from agent_system.contracts import AgentTask, Capability, DispatchRequest, Principal, TaskResult


class RemoteExecutorClient:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._http = http_client
        self._capability_cache: list[Capability] | None = None

    @property
    def name(self) -> str:
        return self._name

    async def capabilities(self) -> list[Capability]:
        if self._capability_cache is not None:
            return list(self._capability_cache)
        response = await self._http.get(f"{self._base_url}/v1/capabilities")
        response.raise_for_status()
        payload = response.json()
        self._capability_cache = [
            Capability.model_validate(item) for item in payload["actions"]
        ]
        return list(self._capability_cache)

    async def execute(self, task: AgentTask, principal: Principal) -> TaskResult:
        request = DispatchRequest(task=task, principal=principal)
        try:
            response = await self._http.post(
                f"{self._base_url}/v1/tasks",
                json=request.model_dump(mode="json"),
                timeout=30.0,
            )
            response.raise_for_status()
            return TaskResult.model_validate(response.json())
        except httpx.HTTPError as exc:
            return TaskResult.failed(
                task.task_id,
                code="EXECUTOR_UNAVAILABLE",
                message=str(exc),
                retryable=True,
            )

