from crs.common.aio import Path
from typing import Optional, Protocol

from opentelemetry import trace
from pydantic import BaseModel, Field

from crs import config
from crs.common.types import POVTarget, Message, Tool, ToolResult, Result, Ok, Err, ToolSuccess, CRSError, Decoder, DecodedPOV, AnalyzedVuln, AgentAction
from crs.common.utils import cached_property, tool_wrap, require, requireable, trim_tool_output
from crs.modules.project import Harness, CrashResult
from crs.modules.debugger import BREAKPOINT_NOT_HIT_ERR
from crs.agents.dynamic_debug import CRSDynamicDebug
from crs.agents.tool_required_agent import ToolRequiredAgent, ToolVerifyClass
from crs.agents.xml_agent import XMLAgent, XMLVerifyClass
from crs.agents.source_questions import CRSSourceQuestions, SourceQuestionsResult

from crs_rust import logger

SUFFICIENT_QUALIFYING_QUERIES = 10

@XMLVerifyClass
class HarnessInputEncoderResult(BaseModel):
    encoder_python: str = Field(description="the python code for the harness input encoder function. note: before returning, you must confirm it works as expected")
    harness_notes: str = Field(description="extra documentation of the harness behavior. anything that is not obvious from the encoder_python should be included here")


class HarnessInputEncoderCallback(Protocol):
    async def __call__(self, harness_num: int) -> Result[HarnessInputEncoderResult]:
        ...

class HarnessInputEncoderAgent(XMLAgent[HarnessInputEncoderResult]):
    @property
    def return_type(self):
        return HarnessInputEncoderResult

    @cached_property
    def tools(self) -> dict[str, Tool]:
        tools: dict[str, Tool] = {
            "read_definition": tool_wrap(self.crs.searcher.read_definition),
            "list_definitions": tool_wrap(self.crs.searcher.list_definitions),
            "read_source": tool_wrap(self.crs.searcher.read_source),
            "find_references": tool_wrap(self.crs.searcher.find_references),
            "query_coverage": tool_wrap(self.query_coverage, post_hooks=[self.terminate_reminder_hook]),
            "get_output": tool_wrap(self.get_output, post_hooks=[self.terminate_reminder_hook]),
        }
        match self.crs.project.info.language:
            case "c"|"c++": tools["gdb_exec"] = tool_wrap(self.gdb_exec, post_hooks=[self.terminate_reminder_hook])
            case "jvm": tools["jdb_exec"] = tool_wrap(self.jdb_exec, post_hooks=[self.terminate_reminder_hook])
            case _: pass
        if self.decoder is not None:
            tools["test_decoding"] = tool_wrap(self.test_decoding)
        return tools

    async def query_coverage(self, input_python: str, target_file: str, target_line: int):
        res = await self.crs.coverage.query_coverage(self.harness_num, input_python, target_file, target_line)
        match res:
            case Ok(cov_res) if cov_res["line_reached"]:
                self.qualifying_test_queries += 1
            case Err(_):
                # reluctantly allow errors to qualify, since it may be impossible to accurately query
                self.qualifying_test_queries += 1
            case _:
                pass
        return res

    async def terminate_reminder_hook[**P, R](self, res: ToolResult[R], input_python: str, *args: P.args, **kwargs: P.kwargs):
        if self.qualifying_test_queries >= SUFFICIENT_QUALIFYING_QUERIES and self.decoder_requirement:
            res.action.append.append(
                Message(
                    role="user",
                    content=(
                        "NOTE: You've made enough qualifying test queries. If you're satisfied with your "
                        "encoding logic, please produce a result. Only if you are seeing unexpected "
                        "results should you continue iterating on your encoder."
                    )
                )
            )
        return res

    @requireable
    async def get_output(self, input_python: str):
        contents = require(await self.crs.project.build_pov(input_python))
        artifacts = require(await self.crs.project.build_default())
        harness = require(await self.crs.project.init_harness_info())[self.harness_num]
        res = await artifacts.run_pov(contents, harness.name)
        return Ok(trim_tool_output(res.output))

    @requireable
    async def gdb_exec(self, input_python: str, source_file: str, line_number: int, commands: str) -> Result[str]:
        breakpoint = f"{Path(source_file).name}:{line_number}"
        input = require(await self.crs.project.build_pov(input_python))
        res = await self.crs.debugger.gdb_exec(self.harness_num, input, breakpoint, commands.splitlines())
        match res:
            case Err(CRSError(error=error)) if error == BREAKPOINT_NOT_HIT_ERR:
                pass
            case _:
                self.qualifying_test_queries += 1
        return res

    @requireable
    async def jdb_exec(self, input_python: str, className: str, line_number: int, commands: str) -> Result[str]:
        breakpoint = f"{className}:{line_number}"
        input = require(await self.crs.project.build_pov(input_python))
        res = await self.crs.debugger.jdb_exec(self.harness_num, input, breakpoint, commands.splitlines())
        match res:
            case Err(CRSError(error=error)) if error == BREAKPOINT_NOT_HIT_ERR:
                pass
            case _:
                self.qualifying_test_queries += 1
        return res

    @requireable
    async def test_decoding(self, input_python: str):
        assert self.decoder is not None
        self.decoder_requirement = True
        input = require(await self.crs.project.build_pov(input_python))
        match await self.decoder.decode_all({"input": input}):
            case Err() as e:
                return e
            case Ok(results):
                return Ok(results["input"])

    # override XMLAgent.get_result
    def get_result(self, msg: Message) -> HarnessInputEncoderResult | Message | None:
        if not msg.tool_calls and (self.qualifying_test_queries == 0 or not self.decoder_requirement):
            return Message(
                role="user",
                content=(
                    "You may not return a result until you have made a qualifying test query and "
                    "tested against the decoder. Note: qualifying queries are coverage or "
                    "debugger queries where you demonstrate the harness if correctly parsing "
                    "the input buffer."
                )
            )
        return super().get_result(msg)

    def __init__(self, crs: 'CRSPovProducerBufGen', harness_num: int, harness: Harness, harness_func_src: Optional[str], tips: str, decoder: Optional[Decoder] = None):
        self.harness_num = harness_num
        self.harness = harness
        self.harness_func_src = harness_func_src
        self.crs = crs
        self.tips = tips
        self.qualifying_test_queries = 0
        self.decoder_requirement = True if decoder is None else False
        self.decoder = decoder
        super().__init__()


class CRSPovProducerBufGen(CRSDynamicDebug):
    async def get_harness_input_encoder(self, harness_num: int) -> Result[HarnessInputEncoderResult]:
        return await self.generate_harness_input_encoder(harness_num)

    @requireable
    async def generate_harness_input_encoder(self, harness_num: int, decoder: Optional[Decoder] = None):
        harnesses = require(await self.project.init_harness_info())
        fixed_harness_num = ""
        if len(harnesses) == 1 and harness_num != 0:
            harness_num = 0
            fixed_harness_num = (
                f"(By the way, you passed in a harness_num of {harness_num} but there is "
                "only 1 harness available and it has id 0. So this is the output for harnes_num = 0)\n"
            )

        harness = require(self.project.check_harness(harness_num))
        tips = await self.harness_tips(harness)

        harness_func_src = await self.project.read_harness_func(harness)
        agent = HarnessInputEncoderAgent(
            crs=self,
            harness_num=harness_num,
            harness=harness,
            harness_func_src=harness_func_src,
            tips=tips,
            decoder=decoder
        )
        res = await agent.run()
        if res.response is None and not res.terminated:
            # ensure it's allowed to terminate
            agent.qualifying_test_queries += 1
            agent.decoder_requirement = True

            # try to get it to terminate in a single iter
            agent.append_user_msg(
                "<important>"
                "Disregard your current thought process. This session is ending. You MAY NOT "
                "make any more queries. You MUST terminate NOW with your best attempt for the result. "
                "</important>"
            )
            res = await agent.run(max_iters=1)
        match res.response:
            case None:
                return Err(CRSError("The agent did not return a result. Please try again."))
            case HarnessInputEncoderResult(encoder_python=encoder_python, harness_notes=harness_notes):
                return Ok(HarnessInputEncoderResult(encoder_python=encoder_python, harness_notes=fixed_harness_num+harness_notes))

@ToolVerifyClass
class POVProducerResult(BaseModel):
    success: bool = Field(description="Whether you succeeded to produce a PoV AND confirmed it works with the test_pov tool. Do not set this without testing the pov first.")
    failure_reason: Optional[str] = Field(default=None, description="If you failed to produce a PoV, provide a brief summary of why. You may not give up without testing some POVs first.")
    reasoning: Optional[str] = Field(default=None, description="Brief reasoning for the outcome, including key constraints or assumptions.")
    generated_input: Optional[str] = Field(default=None, description="The input content or a concise representation of the generated test input that looked promising.")
    test_command: Optional[str] = Field(default=None, description="A concrete command showing how to run the harness with the generated input.")

class ConfirmedPOVProducerResult(POVProducerResult):
    target: POVTarget
    pov_python: str
    crash_result: CrashResult

class CRSPovProducerAgent(ToolRequiredAgent[POVProducerResult]):
    name = "pov_producer"

    @property
    def return_type(self):
        return POVProducerResult

    async def seed_hook[**P](self, harness_num: int, pov_python: str, *args: P.args, **kwargs: P.kwargs):
        if self.crs.project.check_harness(harness_num).is_err():
            return None
        match await self.crs.project.build_pov(pov_python):
            case Err(): return None
            case Ok(contents): pass
        await self.crs.seed_hook(harness_num, contents)

    async def record_result(self, harness_num: int, pov_python: str, crash: CrashResult):
        harnesses = (await self.crs.project.init_harness_info()).unwrap()
        self.result = ConfirmedPOVProducerResult(
            success=True,
            target=POVTarget(
                task_uuid=self.crs.task.task_id,
                project_name=self.crs.project.name,
                harness=harnesses[harness_num].name,
                sanitizer=crash.config.SANITIZER,
                engine=crash.config.FUZZING_ENGINE,
            ),
            pov_python=pov_python,
            crash_result=crash,
        )

    async def test_pov_hook[**P](self, harness_num: int, pov_python: str, *args: P.args, **kwargs: P.kwargs):
        match await self.test_pov(harness_num, pov_python):
            case Ok(CrashResult() as res):
                await self.record_result(harness_num, pov_python, res)
                return ToolSuccess(
                    result=res,
                    action=AgentAction(stop=True)
                )
            case _:
                pass

    async def pov_hook[**P](self, res: ToolResult[CrashResult], harness_num: int, pov_python: str, *args: P.args, **kwargs: P.kwargs):
        if isinstance(res, ToolSuccess):
            await self.record_result(harness_num, pov_python, res.result)
            res.action.stop = True
        else:
            is_python_error = "python" in res.error or "input.bin" in res.error
            if not is_python_error:
                self.failed_pov_count += 1
            if self.failed_pov_count in {6, 12, 18}:
                res.action.append = [Message(
                    role="user",
                    content=(
                        "It looks like you have been having trouble with getting the POV to work. "
                        "Please stop and take a moment to reflect. Try to come up with 3 hypotheses "
                        "for why your PoV might not work. Then try to figure out if you can test if "
                        "any of those hypotheses is correct. If you are making assumptions, attempt "
                        "to ascertain if the assumption really holds.\n"
                        "After reflecting on the above tips, please proceed to produce, test, and debug PoVs. "
                        "You may only give up when you are certain you've proven the bug to be unexploitable. "
                    )
                )]
            if res.extra is None:
                res.extra = {}
            if not is_python_error:
                res.extra['tip'] = (
                    "NOTE: It is likely one of your assumptions is incorrect. For example, the PoV may not be reaching the "
                    "vulnerable code or may not be triggering the vulnerable behavior how you expect. "
                    "You may want to check some of your assumptions using the debug_pov tool."
                )
        return res

    @requireable
    async def test_pov(self, harness_num: int, pov_python: str):
        return await self.crs.task.test_pov(harness_num, pov_python)
    
    async def source_code_questions(self, question: str, additional_info: str = "") -> Result[SourceQuestionsResult]:
        return await self.crs.source_code_questions(question, additional_info, self.rawdiff)

    @cached_property
    def _tools(self) -> dict[str, Tool]:
        tools: dict[str, Tool] = {
            "get_harness_input_encoder": tool_wrap(self.crs.get_harness_input_encoder, pre_hooks=[self.crs.encoder_hook]),
            # "search_call_paths": tool_wrap(self.crs.searcher.search_call_paths),
            # "search_call_paths_from_harness": tool_wrap(self.crs.project.search_call_paths_from_harness),
            "source_questions": tool_wrap(self.source_code_questions),
            "test_pov": tool_wrap(self.test_pov, pre_hooks=[self.seed_hook], post_hooks=[self.pov_hook]),
            "debug_pov": tool_wrap(self.crs.debug_pov, pre_hooks=[self.seed_hook, self.test_pov_hook]),
        }
        # restrict source questions to our known harness starting point
        if self.close_pov:
            tools["source_questions"] = tool_wrap(self.crs.source_code_questions_for_harness(self.close_pov[0].harness))
        if self.crs.project.info.language.lower() == "jvm":
            tools["get_sanitizer_description"] = tool_wrap(self.crs.get_sanitizer_description)
        return tools

    def __init__(
        self,
        crs: 'CRSPovProducer',
        vuln: AnalyzedVuln,
        harnesses: list[Harness],
        close_pov: Optional[tuple[DecodedPOV, str, str]] = None,
        rawdiff: bool = False
    ):
        self.crs = crs
        self.vuln = vuln
        self.harnesses = harnesses
        self.close_pov = close_pov
        self.failed_pov_count = 0
        self.rawdiff = rawdiff
        super().__init__()

class CRSPovProducer(CRSPovProducerBufGen, CRSSourceQuestions):
    async def seed_hook(self, harness_num: int, contents: bytes):
        """
        Called when a potential seed is created; Should be overridden by child classes.
        """
        logger.warning("default seed_hook called, ignoring seed data")

    async def encoder_hook(self, harness_num: int) -> Optional[ToolResult[HarnessInputEncoderResult]]:
        """
        Called when asked to generate a harness input encoder for {harness_num}.
        If it returns non-None, the real tool call is skipped.
        May be overridden by child classes.
        """
        return None

    @config.telem_tracer.start_as_current_span(
        "produce_pov",
        attributes={"crs.action.category": "input_generation", "crs.action.name": "produce_pov"},
        record_exception=False,
    )
    @requireable
    async def produce_pov(
        self,
        vuln: AnalyzedVuln,
        model_idx: int,
        close_pov: Optional[tuple[DecodedPOV, str, str]] = None,
        rawdiff: bool = False,
    ) -> Result[POVProducerResult]:
        """
        Tries to produce a valid proof-of-vulnerability for a given bug.
        `rawdiff` implies the vulnerability is from a DeltaTask raw diff.

        Parameters
        ----------
        vulnerability : str
            A long, detailed description of the vulnerability that was identified.

        Returns
        -------
        dict
            contains "success": bool, "description": str, "pov_id": Optional[int]
        """
        harnesses = require(await self.project.init_harness_info())
        agent = CRSPovProducerAgent(crs=self, vuln=vuln, harnesses=harnesses, close_pov=close_pov, rawdiff=rawdiff)
        agent.model_idx = model_idx
        extra_info: dict[str, str] = {}
        response = (await agent.run()).response
        match response:
            case ConfirmedPOVProducerResult(success=True) as confirmed:
                extra_info["crs.action.target.harness"] = confirmed.target.harness
            case POVProducerResult(success=True):
                response = POVProducerResult(success=False, failure_reason="the agent mistakenly thought it succeeded")
            case None:
                response = POVProducerResult(success=False, failure_reason="the agent did not produce a result")
            case _:
                pass
        span = trace.get_current_span()
        span.add_event(
            f"pov producer complete",
            attributes={"crs.debug.success": response.success, **extra_info}
        )
        return Ok(response)
