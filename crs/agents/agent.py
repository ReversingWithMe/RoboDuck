from abc import ABC, abstractmethod
from base64 import b64encode
from types import NoneType
from typing import Any, Callable, Dict, List, Mapping, Optional, Self, Sequence
import asyncio
import os
import functools
import gzip
import inspect
import textwrap
import time
import typing

import jsonpickle # type: ignore
import orjson

from crs import config
from crs.agents.agent_meta import running_agent, running_tool_call
from crs.agents.crsbase import CRSBase
from crs.common.constants import MAX_TOOL_CALL_RESULT_LENGTH
from crs.common.llm_api import convert_tools, priority_completion, ContextWindowExceeded
from crs.common.workdb import cur_job_priority
from crs.common.types import (
    AgentAction, AgentResult, Message, ReasoningEffort, Tool, ToolCall, ToolResult,
    ToolError, ToolChoice, ModelResponse, Prediction, Priority, Result, Ok, Err, CRSError
)
from crs.common.utils import cached_property, SoloTaskGroup, requireable, require, run_coro_batch
from crs.common.prompts import PromptManager, prompt_manager

from crs_rust import logger

DEFAULT_TEMP = 0.7
TOO_LARGE_MESSAGE = orjson.dumps({
    "error": "Tool call output was too large to return. Please try another approach."
})

def get_model_options(agent_name: str):
    model_map = config.MODEL_MAP.get()
    return model_map.get(agent_name, [config.MODEL.get()])

def type_repr(typ: type) -> str:
    origin = typing.get_origin(typ)
    if origin is typing.Union:
        args = set(typing.get_args(typ)) - {NoneType}
        if len(args) != 1:
            raise TypeError(f"unsupported Union: {typ}")
        inner = args.pop()
        return type_repr(inner)
    if typ is int:
        return "int"
    if typ is str:
        return "str"
    if typ is bool:
        return "bool"
    raise TypeError(f"unsupported tool arg type: {typ}")

class AgentGeneric[T](ABC):
    @cached_property
    def tools(self) -> Mapping[str, Tool]:
        return {}

    @cached_property
    def _tools_api(self) -> Optional[list[dict[str, Any]]]:
        tools = self.tools
        orig_docs = {name: tool.__doc__ for name, tool in tools.items()}
        # TODO: can/should this be cached?
        try:
            for name, tool in tools.items():
                if tool.__doc__ is not None:
                    if name in self.prompts.tools:
                        if name == "terminate":
                            tool.__doc__ = f"{self.prompts.tools[name].summary}\n\n{tool.__doc__}"
                        else:
                            raise RuntimeError(f"{self.__class__.__name__} tool {name!r} has docstring but is also in prompt yaml")
                    continue
                prompt = self.prompts.tools[name]
                sig = inspect.signature(tool)
                params = "\n".join(
                    f"{name} : {type_repr(param.annotation)}\n{textwrap.indent(prompt.params[name], '    ')}"
                    for name, param in sig.parameters.items()
                )
                chunks: list[str] = [f"{prompt.summary}\n"]
                if prompt.params:
                    chunks.append(f"Parameters\n----------\n{params}")
                if prompt.returns:
                    chunks.append(f"\nReturns\n-------\nresult\n{textwrap.indent(prompt.returns, '    ')}")
                tool.__doc__ = "\n".join(chunks)
            result = convert_tools(tools)
            return result
        finally:
            for name, tool in tools.items():
                tool.__doc__ = orig_docs[name]

    @property
    def model(self) -> str:
        model_options = get_model_options(self.__class__.__name__)
        return model_options[self.model_idx % len(model_options)]

    @property
    def temperature(self) -> float:
        return DEFAULT_TEMP

    @property
    def tool_choice(self) -> Optional[ToolChoice]:
        return "auto" if self.tools else None

    @property
    def reasoning_effort(self) -> Optional[ReasoningEffort]:
        if "o4-mini" in self.model or "o3-mini" in self.model or "o1-mini" in self.model:
            return "high"
        if "o1" in self.model or "o3" in self.model:
            return "medium"
        if "claude" in self.model and "thinking" in self.model:
            if "low" in self.model:
                return "low"
            if "medium" in self.model:
                return "medium"
            if "high" in self.model:
                return "high"
            return "low"
        return None

    @property
    def use_caching(self) -> bool:
        return True

    @property
    def prediction(self) -> Optional[Prediction]:
        return None

    @property
    def priority(self) -> float:
        return cur_job_priority.get(Priority.LOWEST)

    @property
    def logprobs(self) -> Optional[bool]:
        return None

    @property
    def top_logprobs(self) -> Optional[int]:
        return None

    @property
    def max_completion_tokens(self) -> Optional[int]:
        value = os.getenv("MAX_COMPLETION_TOKENS")
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @property
    def logit_bias(self) -> Optional[dict[int, float]]:
        return None

    @property
    def init_msgs(self) -> bool:
        return True

    @property
    def compressible(self) -> bool:
        return True

    def mock_response(self, msgs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return None

    @classmethod
    def prompt_manager(cls) -> PromptManager:
        return prompt_manager

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.parent = running_agent.get()
        self.msgs: List[Dict[str, Any]] = []
        self.terminated = False
        self.cost = 0.0
        self.tool_errors = 0
        self.id = f"{id(self)}_{time.time()}"
        self._log_creation()
        self.model_idx = 0

        names: list[str] = []
        bases = [self.__class__]
        while bases:
            names += [cls.__name__ for cls in bases]
            bases: list[type[Any]] = [inner
                     for outer in bases
                         for inner in outer.__bases__
                             if issubclass(inner, AgentGeneric)]

        self.prompts = self.prompt_manager().model(self.model).bind(*names, kwargs={"agent": self})
        if self.init_msgs:
            self._append_msg(Message(role="system", content=self.prompts.system))
            self._append_msg(Message(role="user", content=self.prompts.user))

    def _log_creation(self) -> None:
        logger.info("Creating agent '{name}' with id={agent}", name=self.__class__.__name__, agent=self.id)

    def append_user_msg(self, msg: str):
        self._append_msg(Message(role="user", content=msg))

    def _append_msg(self, msg: Message):
        logger.info(
            "Appending message to agent",
            agent=self.id,
            pre_serialized=self.serialize_compressed() if config.SERIALIZE_AGENTS and msg.role == "assistant" else None,
            total_cost=self.cost,
            **msg.model_dump()
        )
        self.msgs.append(msg.model_dump())

    def _add_cost(self, cost: float):
        self.cost += cost
        if self.parent is not None:
            self.parent._add_cost(cost)

    def _add_tool_error(self):
        self.tool_errors += 1
        if self.parent is not None:
            self.parent._add_tool_error()

    async def _do_tool_call(self, tool_name: str, tool_call_id: str, tool: Tool, **kwargs: Dict[str, Any]) -> ToolResult[Any]:
        # create a new task (with new context) for each tool call
        # encode the args roughly for telemetry
        encoded_args: dict[str, str] = {}
        for k,v in kwargs.items():
            encoded_args[f"crs.debug.build.kwarg.{k}"] = repr(v)
        with config.telem_tracer.start_as_current_span(
            tool_name,
            attributes=encoded_args,
            record_exception=False,
        ) as _:
            _ = running_tool_call.set(tool_call_id)
            return await tool(**kwargs)

    async def _handle_tool_call(self, tool_call: ToolCall) -> Optional[AgentAction]:
        if self.terminated:
            logger.warning("Skipping tool call because agent is terminated")
            self._append_msg(Message(
                tool_call_id=tool_call.id,
                role="tool",
                name="cancelled",
                content="Tool call was canceled by admin request",
            ))
            return None

        try:
            kwargs: dict[str, Any] = await asyncio.to_thread(orjson.loads, tool_call.function.arguments)
            kwargs.pop('', None) # remove any kwargs with no name; sometimes o3-mini generates these
        except orjson.JSONDecodeError:
            self._add_tool_error()
            logger.warning("Tool call arguments were invalid JSON")
            tool_call.function.name = "ERROR"
            self._append_msg(Message(
                tool_call_id=tool_call.id,
                role="tool",
                name="ERROR",
                content="ERROR: failed to decode tool arguments as proper JSON",
            ))
            return None
        tool_name = tool_call.function.name
        if (fn := self.tools.get(tool_name)) is None:
            self._add_tool_error()
            logger.warning("Tool call function name {tool_name} is not defined", tool_name=tool_name)
            tool_call.function.name = "ERROR"
            available_summary = ",".join(self.tools.keys())
            self._append_msg(Message(
                tool_call_id=tool_call.id,
                role="tool",
                name="ERROR",
                content=f"ERROR: {tool_name} NOT DEFINED. Available tools: {available_summary}",
            ))
            return None

        # attempt to recover from tool exceptions
        try:
            logger.info("Making tool call: {tool_name}", tool_name=tool_name)
            async with SoloTaskGroup() as tg:
                res = await tg.create_task(
                    self._do_tool_call(tool_name, tool_call.id, fn, **kwargs),
                    name=f"agent {self.__class__.__name__} _do_tool_call(name={tool_name!r})",
                )
        except Exception as e:
            self._add_tool_error()
            logger.exception("tool call encountered unexpected error")
            res = ToolError(repr(e))
            tool_name = "ERROR"

        result = res.model_dump()
        result.pop("action", None)
        result = await asyncio.to_thread(orjson.dumps, result)
        if len(result) > MAX_TOOL_CALL_RESULT_LENGTH:
            result = TOO_LARGE_MESSAGE
        self._append_msg(Message(
            tool_call_id=tool_call.id,
            role="tool",
            name=tool_name,
            content=result.decode(),
        ))

        return res.action

    @abstractmethod
    def get_result(self, msg: Message) -> T | Message | None:
        pass

    async def _compress_context(self) -> Result[None]:
        # TODO: can we have a model summarize the context instead?
        # i.e. actually 'compress' the context instead of 'truncate' it
        logger.info("compressing context...")
        pre = self.msgs[:2]
        pre.append({'role': 'user', 'content': '[some messages were removed due to context size constraints]'})
        post = self.msgs[2*len(self.msgs)//3:]
        while post and post[0]['role'] != 'assistant':
            _ = post.pop(0)
        if not post:
            return Err(CRSError("Not enough messages to preserve an assistant message"))
        msgs = pre + post
        self.msgs = msgs
        return Ok(None)

    async def _completion(self, n: int = 1) -> Result[ModelResponse]:
        assert self.msgs[-1]["role"] != "assistant", "Invalid message state"
        return await priority_completion(
            self.priority,
            n=n,
            model=self.model,
            messages=self.msgs,
            temperature=self.temperature,
            tools=self._tools_api,
            tool_choice=self.tool_choice,
            use_caching=self.use_caching,
            prediction=self.prediction,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            max_completion_tokens=self.max_completion_tokens,
            reasoning_effort=self.reasoning_effort,
            mock_response=self.mock_response(self.msgs),
            logit_bias=self.logit_bias
        )

    @requireable
    async def completion(self, n: int = 1):
        match (response := await self._completion(n=n)):
            case Err(ContextWindowExceeded()) if self.compressible:
                require(await self._compress_context())
                response = await self._completion(n=n)
            case Err(_):
                # another error occured during completion (even after the internal retry logic)
                # try to fallback to another model (if configured)
                self.model_idx += 1
                response = await self._completion(n=n)
            case _:
                pass
        match response:
            case Ok(resp):
                # if the completion was a success, add to the cost
                self._add_cost(resp.cost)
            case _:
                pass
        return response

    async def _iter(self) -> Optional[T]:
        model_response = (await self.completion()).unwrap() # no choice but to unwrap()
        msg = model_response.choices[0].message
        self._append_msg(msg)

        tool_calls = msg.tool_calls or []
        handle_tools = [self._handle_tool_call(tool_call) for tool_call in tool_calls]
        actions = await asyncio.gather(*handle_tools)
        for action in actions:
            if action is None:
                continue
            if action.stop:
                logger.info("Stopping agent due to AgentAction")
                self.terminated = True
            if action.append:
                logger.info("Appending messages due to AgentAction")
                for new_msg in action.append:
                    self._append_msg(new_msg)
            if action.rewind:
                logger.error("AgentAction.rewind not implemented yet")
                raise NotImplementedError

        match self.get_result(msg):
            case None:
                pass
            case Message() as m:
                self._append_msg(m)
            case r:
                self.terminated = True
                return r

    async def _run(self, max_iters: int = 30) -> AgentResult[T]:
        self.terminated = False
        _ = running_agent.set(self) # set the running agent for the current context
        logger.info(
            "Running agent {name} for <= {max_iters} iters",
            name=self.__class__.__name__,
            max_iters=max_iters
        )
        response = None
        for _ in range(max_iters):
            response = await self._iter()
            if self.terminated or response is not None:
                break
        return AgentResult(response=response, terminated=self.terminated, msgs=self.msgs)

    async def run(self, max_iters: int = 40) -> AgentResult[T]:
        # create a new task (with new context) for each agent run
        async with self.lock, SoloTaskGroup() as tg:
            return await tg.create_task(self._run(max_iters=max_iters), name=f"agent {self.__class__.__name__}.run()")

    # type ignore justification: classmethod typing doesn't accept our cls type annotation,
    # but annotating cls this way is required to allow type inference of P at the call sites
    @classmethod # type: ignore
    async def run_default_batch[**P](
        cls: Callable[P, 'AgentGeneric[T]'],
        stop_condition: Optional[Callable[[T], bool]],
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Sequence[T]:
        """
        Runs the default batch of agents (one per configured model) aggregates the results.
        The batch contains one agent per configured model in the {MODEL_MAP} for {cls.__name__}.
        Use {stop_condition} if you want to cancel other agents once a qualifying result is produced.
        Returns all non-None responses.
        """
        num_models = len(get_model_options(cls.__name__))
        agents: list[AgentGeneric[T]] = []
        for i in range(num_models):
            agent = cls(*args, **kwargs)
            agent.model_idx = i
            agents.append(agent)

        return await run_coro_batch(
            [agent.run() for agent in agents],
            name=f"run_default_batch() {agents=}",
            filter=lambda res: res.response,
            stop_condition=stop_condition
        )

    def fork(self: Self):
        return self.__class__.deserialize(self.serialize())

    @property
    def serialize_overrides(self) -> dict[str, Callable[[Self], Any]]:
        return {"parent": lambda _: None, "lock": lambda _: asyncio.Lock()}

    def __getstate__(self) -> object:
        res = self.__dict__.copy()
        for key in self.serialize_overrides:
            del res[key]
        excludes = {
            key for key in res.keys()
            if isinstance(getattr(self.__class__, key, None), functools.cached_property)
        }
        for key in excludes:
            del res[key]
        return res

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__ = state
        for key, factory in self.serialize_overrides.items():
            self.__dict__[key] = factory(self)

        # emit logs for the agent
        self._log_creation()
        self.msgs, msgs = [], self.msgs
        for msg in msgs:
            self._append_msg(Message(**msg))

    def serialize_compressed(self) -> str:
        return b64encode(gzip.compress(self.serialize().encode())).decode()

    def serialize(self) -> str:
        return jsonpickle.encode(self) # type: ignore

    @classmethod
    def deserialize(cls, serialized: str, **properties: Any):
        res = jsonpickle.decode(serialized) # type: ignore
        assert isinstance(res, cls), "serialized value is not an Agent"
        # set any additional properties
        for key, value in properties.items():
            setattr(res, key, value)
        return res

class Agent(AgentGeneric[str]):
    def get_result(self, msg: Message) -> str | Message | None:
        if not msg.tool_calls:
            if msg.content is None:
                return Message(role='user', content='If you are done, please return your results')
            return msg.content
        return None

class CRSAgent[T, U: CRSBase](AgentGeneric[T]):
    def __init__(self, crs: U) -> None:
        self.crs = crs
        super().__init__()


class MsgHistoryAgent(Agent):
    """
    An agent which has another agent's message history, but may have a new set of tools
    """
    def __init__(self, msgs: List[Dict[str, Any]]) -> None:
        super().__init__()
        for msg in msgs:
            self._append_msg(Message(**msg))
        self._append_msg(Message(role="user", content=self.prompts.user))

    @cached_property
    def tools(self) -> dict[str, Tool]:
        return {}

    @property
    def init_msgs(self):
        return False
