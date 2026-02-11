"""Run only the AI/static analysis (stage 2) and scoring (stage 3) on a local repo snapshot.

First, a lightweight "stage 0" indexing step writes a file inventory under config.CACHE_DIR/stage023-ingest for repeatability. Set `--skip-ingest` to run directly on the directory and `--cache-dir`/`--ingest-exclude` to control where/what gets indexed.

Docker / runtime notes:
- Pull `python:3.12-slim` (or similar) so you have a clean, portable environment that matches the repo's
  Python 3.12 requirement. Example:
  ```sh
  docker pull python:3.12-slim
  docker run --rm -it -v "$PWD":/workspace -w /workspace python:3.12-slim /bin/bash
  ```
- Inside the container install the Rust/Python extension and dependencies before running this script:
  ```sh
  python -m pip install -e .
  ```
- You also need the tree-sitter language bindings used by the analyzers (`tree-sitter`, `tree-sitter-c`,
  `tree-sitter-cpp`, and `tree-sitter-java`). The `pyproject.toml` already pins them, so the editable
  install above pulls what you need. If you prefer a second image for offline tree parsing, the same
  `python:3.12-slim` container works once the wheels build.
- Because stage 2/3 lean on `litellm`, you must provide API credentials (OpenAI/Anthropic/Gemini/Azure)
  via environment variables; without them, the LLM calls will fail, so you cannot run this fully offline.

This script does not attempt to boot the full CRS task/submitter stack; it simply builds an `AnalysisProject` from
a directory, runs the single-function and/or multi-function LLM analysers, and then scores the resulting vulnerability
descriptions with the `LikelyVulnClassifier` (stage 3). You can point it at any checkout of the repository.

You can place LLM credentials in a `.env` file so that `litellm` picks them up when the script starts.
For example:

```
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
LITELLM_ENDPOINT=https://api.litellm.com/v1
LITELLM_API_KEY=...
MODEL_MAP=/workspace/configs/models-final.toml
```

The script accepts `--env-file` if you store secrets elsewhere.

You can build the provided `Dockerfile` with `docker build -t stage023-runner .` and then pass the host repo and `.env` into the container:

```sh
docker build -t stage023-runner .
docker run --rm -it -v "$PWD":/workspace -v "$PWD/.env":/.env stage023-runner /workspace --mode both
```
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import re
import json
import os
import pathlib
import subprocess
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, TypedDict, cast, Optional
import typing

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DEFAULT_BASE_ENV_PATHS = (
    os.getenv("STAGE023_ENV_FILE"),
    "/.env",
    "./.env",
)

_ = os.environ.setdefault("LOG_LEVEL", "WARNING")
_ = os.environ.setdefault("LITELLM_LOG", "ERROR")


def load_env_from_file(path: pathlib.Path, silent: bool = False) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        _ = os.environ.setdefault(key, value)
        loaded[key] = value
    if loaded and not silent:
        print(f"loaded {len(loaded)} vars from {path}")
    return loaded


def progress_bar(total: int, enabled: bool, label: str):
    if not enabled:
        return None
    if tqdm is None:
        return None
    return tqdm(total=total, desc=label, unit="step")


async def run_testflight(model: str) -> None:
    messages = [
        {"role": "system", "content": "You are a testflight probe."},
        {"role": "user", "content": "Respond with OK."},
    ]
    result = await completion(model=model, messages=messages, tool_choice="none")
    try:
        _ = result.unwrap()
    except Exception as exc:
        raise SystemExit(f"Testflight failed for model {model}: {exc}")


for candidate in DEFAULT_BASE_ENV_PATHS:
    if not candidate:
        continue
    env_path = pathlib.Path(candidate)
    if env_path.exists():
        _ = load_env_from_file(env_path, silent=True)
        break

from crs import config
from crs.agents.triage import DedupClassifier
from crs.agents.vuln_analyzer import LikelyVulnClassifier, CRSVulnAnalyzerAgent, VulnAnalysis
from crs.agents.pov_producer import CRSPovProducerAgent
from crs.agents.source_questions import SourceQuestionsResult
from crs.analysis import c_tree_sitter, java_tree_sitter
from crs.analysis.data import AnalysisProject, AnnotatedReport, SourceFile, SourceMember
from crs.analysis.full import analyze_project, analyze_project_multifunc
from crs.common.llm_api import completion
from crs.common.prompts import prompt_manager
from crs.common.types import (
    AnalyzedVuln,
    VulnReport,
    LineDefinition,
    FileReferences,
    FileReference,
    Result,
)
from crs.common.core import CRSError, Ok, Err
from crs.common.utils import tool_wrap, cached_property
from crs.modules.project import Harness, HarnessType


class Stage2Record(TypedDict):
    stage: str
    model: str | None
    function: str
    file: str
    description: str
    summary: str
    code_snippet: str
    report: dict[str, object]


class Stage3Score(TypedDict):
    index: int
    function: str
    file: str
    description: str
    stage: str
    avg_likely: float
    max_likely: float
    std_likely: float


class Stage3Trace(TypedDict):
    function: str
    file: str
    description: str
    likely_confidence: float
    likely_above_threshold: bool
    dedupe_choice: str
    dedupe_confidence: float
    triage_note: str
    pov_note: str
    vuln_analysis: dict[str, object]


@dataclass
class FakeHarness:
    name: str
    source: str


@dataclass
class FakeProject:
    name: str
    info: "FakeProjectInfo"


@dataclass
class FakeProjectInfo:
    language: str = "c"


class FakeSearcher:
    def __init__(self, analysis_project: AnalysisProject):
        self.analysis_project = analysis_project
        self._resolve_tool_annotations()

    def _resolve_tool_annotations(self) -> None:
        def resolve(fn: Any) -> None:
            func = fn.__func__ if hasattr(fn, "__func__") else fn
            try:
                func.__annotations__ = typing.get_type_hints(func)
            except Exception:
                pass

        resolve(self.list_definitions)
        resolve(self.read_definition)
        resolve(self.read_source)
        resolve(self.find_references)

    def _resolve_file(self, file_name: str) -> Optional[SourceFile]:
        if file_name in self.analysis_project.files:
            return self.analysis_project.files[file_name]
        for path, sf in self.analysis_project.files.items():
            if path.endswith(file_name):
                return sf
        return None

    def _member_line(self, member: SourceMember) -> int:
        return member.file.offset_to_line(member.range.a) + 1

    def _member_range_lines(self, member: SourceMember) -> tuple[int, int]:
        body_range = getattr(member, "body", member.range)
        start, end = member.file.range_to_lines(body_range)
        return start + 1, end + 1

    async def list_definitions(self, path: str) -> Result[list[LineDefinition]]:
        sf = self._resolve_file(path)
        if sf is None:
            return Err(CRSError("file does not exist"))
        defs: list[LineDefinition] = []
        for member in self.analysis_project.decls:
            if member.file.path != sf.path:
                continue
            name = member.name.decode(errors="replace")
            defs.append(LineDefinition(name=name, line=self._member_line(member)))
        if not defs:
            return Err(CRSError("no results found"))
        return Ok(defs)

    async def read_definition(
        self,
        name: str,
        path: Optional[str] = None,
        line_number: Optional[int] = None,
        display_lines: bool = True,
    ) -> Result[dict[str, object]]:
        _ = line_number, display_lines
        candidates = []
        for member in self.analysis_project.decls:
            if member.name.decode(errors="replace") != name:
                continue
            if path and member.file.path != path and not member.file.path.endswith(path):
                continue
            candidates.append(member)
        if not candidates:
            return Err(CRSError("no results found"))
        member = candidates[0]
        start_line, end_line = self._member_range_lines(member)
        start_offset = member.file.line_index[start_line - 1]
        end_offset = member.file.line_index[end_line] if end_line < len(member.file.line_index) else len(member.file.source)
        contents = member.file.source[start_offset:end_offset].decode(errors="replace")
        return Ok({
            "contents": contents,
            "line_start": start_line,
            "line_end": end_line,
            "file": member.file.path,
        })

    async def read_source(
        self,
        file_name: str,
        line_number: Optional[int] = None,
        line_range: Optional[str] = None,
    ) -> Result[dict[str, object]]:
        sf = self._resolve_file(file_name)
        if sf is None:
            return Err(CRSError("file does not exist"))
        if line_range is not None:
            start_line = None
            end_line = None
            if isinstance(line_range, str):
                parts = [p for p in re.split(r"[,:\s]+", line_range) if p]
                if len(parts) >= 2:
                    try:
                        start_line = int(parts[0])
                        end_line = int(parts[1])
                    except ValueError:
                        start_line = None
                        end_line = None
            elif isinstance(line_range, (tuple, list)) and len(line_range) >= 2:
                try:
                    start_line = int(line_range[0])
                    end_line = int(line_range[1])
                except (TypeError, ValueError):
                    start_line = None
                    end_line = None

            if start_line is None or end_line is None:
                return Err(CRSError("line_range must be two integers"))

            start_line = max(1, start_line)
            end_line = min(len(sf.line_index) - 1, end_line)
            start_offset = sf.line_index[start_line - 1]
            end_offset = sf.line_index[end_line] if end_line < len(sf.line_index) else len(sf.source)
            contents = sf.source[start_offset:end_offset].decode(errors="replace")
            return Ok({
                "contents": contents,
                "line_start": start_line,
                "line_end": end_line,
            })

        if line_number is None:
            return Err(CRSError("line_number is required when line_range is not provided"))

        line_idx = max(0, line_number - 1)
        start_line = max(0, line_idx - 3)
        end_line = min(len(sf.line_index) - 2, line_idx + 3)
        start_offset = sf.line_index[start_line]
        end_offset = sf.line_index[end_line + 1] if end_line + 1 < len(sf.line_index) else len(sf.source)
        contents = sf.source[start_offset:end_offset].decode(errors="replace")
        return Ok({
            "contents": contents,
            "line_start": start_line + 1,
            "line_end": end_line + 1,
        })

    async def find_references(
        self,
        name: str,
        path: Optional[str] = None,
        case_insensitive: bool = False,
    ) -> Result[list[FileReferences]]:
        refs: list[FileReferences] = []
        if case_insensitive:
            needle = name.lower()
        else:
            needle = name
        for sf in self.analysis_project.files.values():
            if path and sf.path != path and not sf.path.endswith(path):
                continue
            file_refs: list[FileReference] = []
            text = sf.source.decode(errors="replace")
            for idx, line in enumerate(text.splitlines(), start=1):
                hay = line.lower() if case_insensitive else line
                if needle not in hay:
                    continue
                enclosing = ""
                for member in self.analysis_project.decls:
                    if member.file.path != sf.path:
                        continue
                    start_line, end_line = self._member_range_lines(member)
                    if start_line <= idx <= end_line:
                        enclosing = member.name.decode(errors="replace")
                        break
                file_refs.append(FileReference(line=idx, content=line.strip(), enclosing_definition=enclosing))
                if len(file_refs) >= 50:
                    break
            if file_refs:
                refs.append(FileReferences(file_name=sf.path, refs=file_refs))
        if not refs:
            return Err(CRSError("no results found"))
        return Ok(refs)


@dataclass
class FakeCRS:
    project: FakeProject
    harnesses: list[FakeHarness]
    root_dir: pathlib.Path
    analysis_project: AnalysisProject
    searcher: FakeSearcher

    @property
    def harness_paths_str(self) -> str:
        return "\n".join(
            f"<harness><num>{i}</num><name>{h.name}</name><source>{h.source}</source></harness>"
            for i, h in enumerate(self.harnesses)
        )

    def harness_path_str(self, harness_name: str) -> str:
        return "\n".join(
            f"<harness><num>{i}</num><name>{h.name}</name><source>{h.source}</source></harness>"
            for i, h in enumerate(self.harnesses)
            if h.name == harness_name
        )

    def trigger_tips(self, get_sanitizer_description_available: bool = False) -> str:
        _ = get_sanitizer_description_available
        return ""


@dataclass
class FakePovCRS(FakeCRS):
    current_file: Optional[pathlib.Path] = None

    async def source_code_questions(self, question: str, additional_info: str = "", rawdiff: bool = False):
        _ = rawdiff
        note = additional_info.strip()
        extra = f" Additional info: {note}" if note else ""
        snippet = await asyncio.to_thread(self._answer_from_sources, question)
        if snippet:
            return Ok(SourceQuestionsResult(answer=f"{snippet}{extra}"))
        return Ok(SourceQuestionsResult(answer=f"No match found for: {question}.{extra}"))

    async def source_code_questions_no_rawdiff(self, question: str, additional_info: str = ""):
        return await self.source_code_questions(question, additional_info, rawdiff=False)

    def source_code_questions_for_harness(self, harness_name: str):
        async def source_code_questions(question: str, additional_info: str = ""):
            _ = harness_name
            return await self.source_code_questions(question, additional_info, rawdiff=False)

        return source_code_questions

    def _answer_from_sources(self, question: str) -> str:
        tokens = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question) if len(t) > 2]
        tokens = [t for t in tokens if t not in {"function", "file", "path", "line", "lines"}]
        if not tokens:
            return ""

        members = self.analysis_project.decls
        member_map: dict[str, list[SourceMember]] = {}
        for member in members:
            name = member.name.decode(errors="replace")
            member_map.setdefault(name, []).append(member)

        answers: list[str] = []
        for token in tokens:
            if token in member_map:
                for member in member_map[token][:3]:
                    body_range = getattr(member, "body", member.range)
                    snippet = member.file[body_range].decode(errors="replace")
                    answers.append(
                        f"Definition of {token} in {member.file.path}:\n{snippet}"
                    )

        # Basic call-site scan
        for token in tokens:
            call_hits: list[str] = []
            for sf in self.analysis_project.files.values():
                text = sf.source.decode(errors="replace")
                for i, line in enumerate(text.splitlines(), start=1):
                    if f"{token}(" in line:
                        call_hits.append(f"{sf.path}:{i}: {line.strip()}")
                        if len(call_hits) >= 5:
                            break
                if len(call_hits) >= 5:
                    break
            if call_hits:
                answers.append(
                    f"Call sites for {token}:\n" + "\n".join(call_hits)
                )

        if answers:
            return "\n\n".join(answers)

        return ""

class LimitedPovProducerAgent(CRSPovProducerAgent):
    @cached_property
    def _tools(self):
        def resolve_annotations(fn: Any) -> Any:
            func = fn.__func__ if hasattr(fn, "__func__") else fn
            try:
                func.__annotations__ = typing.get_type_hints(func)
            except Exception:
                pass
            return fn

        tools = {
            "source_questions": tool_wrap(resolve_annotations(self.crs.source_code_questions_no_rawdiff)),
        }
        return tools


@dataclass
class FakePovAgentContext:
    crs: FakeCRS
    vuln: AnalyzedVuln
    close_pov: Optional[tuple[str, str, str]] = None


async def run_prompt(agent_name: str, model: str, agent_context: object) -> str:
    bound = prompt_manager.model(model).bind(agent_name, kwargs={"agent": agent_context})
    messages = [
        {"role": "system", "content": bound.system},
        {"role": "user", "content": bound.user},
    ]
    print("first", messages)
    result: Any = await completion(model=model, messages=messages, tool_choice="none")
    try:
        response: Any = result.unwrap()
        content = response.choices[0].message.content
        print("second", content)
        return content or ""
    except Exception:
        err_msg = "unknown error"
        if hasattr(result, "unwrap_err"):
            try:
                err: Any = result.unwrap_err()
                err_msg = getattr(err, "error", repr(err))
            except Exception:
                pass
        return f"<error>{err_msg}</error>"


async def run_pov_agent(crs: FakePovCRS, vuln: AnalyzedVuln, model_idx: int = 0) -> str:
    harnesses = [Harness(
        name=crs.harnesses[0].name,
        type=HarnessType.LIBFUZZER,
        source=crs.harnesses[0].source,
        options="",
        harness_func=None,
    )]
    agent = LimitedPovProducerAgent(crs=crs, vuln=vuln, harnesses=harnesses, close_pov=None, rawdiff=False)
    agent.model_idx = model_idx
    agent.append_user_msg(
        "<important>In stage033_runner we cannot execute harness binaries. "
        "Only use source_questions, then terminate with your best guess. "
        "Do not call test_pov/debug_pov and do not assume get_harness_input_encoder is available. "
        "Focus on generating a concrete test command or a clear input-generation recipe that I can run later. "
        "Your SUCCESS should mean you produced a usable test command or recipe. "
        "Your FAILURE should explain what missing information blocks that. "
        "Always include your reasoning and the generated input in the terminate() call using the fields: reasoning, generated_input, test_command. "
        "Also include a <cmd>...</cmd> block showing the command you would run to test the POV if execution were enabled. "
        "You may use a python -c one-liner to generate input.bin instead of embedding large blobs.</important>"
    )
    res = await agent.run(max_iters=2)
    if res.response is None and not res.terminated:
        agent.append_user_msg(
            "<important>Terminate now with your best guess. You cannot call any more tools.</important>"
        )
        res = await agent.run(max_iters=1)
    if res.response is None:
        return "<error>pov agent did not return a response</error>"
    try:
        return json.dumps(res.response.model_dump(), indent=2)
    except Exception:
        return repr(res.response)


def _analysis_summary(result: Optional[VulnAnalysis], error: Optional[str]) -> dict[str, object]:
    if result is None:
        return {
            "triggerable": None,
            "positive": None,
            "negative": error or "analyzer returned no result",
        }
    positive = result.positive.model_dump() if result.positive is not None else None
    return {
        "triggerable": result.triggerable,
        "positive": positive,
        "negative": result.negative,
    }


def _merge_pov_note(pov_note: str, analysis_payload: dict[str, object]) -> str:
    try:
        data = json.loads(pov_note)
        if isinstance(data, dict):
            data["analysis"] = analysis_payload
            return json.dumps(data, indent=2)
    except Exception:
        pass
    return f"{pov_note}\n\n<analysis>\n{json.dumps(analysis_payload, indent=2)}\n</analysis>"


async def run_vuln_analyzer(crs: FakeCRS, record: Stage2Record) -> tuple[Optional[VulnAnalysis], Optional[str]]:
    report = VulnReport(
        task_uuid=uuid.uuid4(),
        project_name=crs.project.name,
        function=record["function"],
        file=record["file"],
        description=record["description"],
    )
    for member in crs.analysis_project.decls:
        if member.name.decode(errors="replace") != record["function"]:
            continue
        if member.file.path != record["file"] and not member.file.path.endswith(record["file"]):
            continue
        start_line, end_line = member.file.range_to_lines(getattr(member, "body", member.range))
        report.function_range = (start_line + 1, end_line + 1)
        break

    agent = CRSVulnAnalyzerAgent(crs=crs, report=report)
    res = await agent.run(max_iters=30)
    if res.response is None and not res.terminated:
        agent.append_user_msg(
            "You have been working for a long time. Please think carefully about how to "
            "finish your analysis quickly. Make no more than 3 more tool calls before "
            "producing your final output."
        )
        res = await agent.run(max_iters=10)
    if res.response is None and not res.terminated:
        agent.append_user_msg(
            "<important>Disregard your current thought process. This session is ending. "
            "You MAY NOT make any more queries. You MUST terminate NOW with your best guess for the result."
            "</important>"
        )
        res = await agent.run(max_iters=1)

    if res.response is None:
        return None, "vuln analyzer did not return a response"
    return res.response, None


def scan_source_files(src_dir: pathlib.Path) -> Iterator[pathlib.Path]:
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if "test" in path.parts or "tests" in path.name:
            continue
        if path.suffix.lower() in (".c", ".java"):
            yield path


async def parse_source_file(path: pathlib.Path) -> tuple[SourceFile, list[SourceMember]]:
    source = path.read_bytes()
    sf = SourceFile(str(path), source)
    parser = c_tree_sitter.parse if path.suffix.lower() == ".c" else java_tree_sitter.parse
    decls = await asyncio.to_thread(parser, sf)
    return sf, decls


async def build_analysis_project(src_dir: pathlib.Path, use_progress: bool) -> AnalysisProject:
    project = AnalysisProject()
    paths = list(scan_source_files(src_dir))
    iterator: Iterable[pathlib.Path] = paths
    if use_progress and tqdm is not None:
        iterator = tqdm(paths, desc="Stage 1 parse", total=len(paths), unit="file", leave=True)
    for path in iterator:
        sf, decls = await parse_source_file(path)
        project.files[str(path)] = sf
        project.decls.extend(decls)
    project.build_lut()
    return project


DEFAULT_INGEST_EXCLUDES = {".git", "__pycache__", "tests", "test"}
DEFAULT_CACHE_ROOT = pathlib.Path(os.fspath(config.CACHE_DIR)) / "stage033-ingest"

async def compute_project_hash(project_dir: pathlib.Path) -> str:
    def _inner() -> str:
        if (project_dir / ".git").is_dir():
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_dir)
                diff = subprocess.check_output(["git", "diff"], cwd=project_dir)
                h = hashlib.sha256()
                h.update(commit)
                h.update(diff)
                return h.hexdigest()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        stat = project_dir.stat()
        h = hashlib.sha256()
        h.update(project_dir.as_posix().encode())
        h.update(str(stat.st_mtime_ns).encode())
        return h.hexdigest()

    return await asyncio.to_thread(_inner)


async def stage0_index(
    project_dir: pathlib.Path,
    cache_root: pathlib.Path,
    project_hash: str,
    excludes: set[str],
    use_progress: bool,
) -> pathlib.Path:
    dest = cache_root / project_hash / project_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    index_path = dest / "index.json"
    if index_path.exists():
        return index_path

    def _build_index() -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        progress = None
        if use_progress and tqdm is not None:
            progress = tqdm(desc="Stage 0 index", unit="file", leave=True)
        for root, dirs, files in os.walk(project_dir):
            rel_root = pathlib.Path(root).relative_to(project_dir)
            dirs[:] = [d for d in dirs if d not in excludes]
            for name in files:
                if name in excludes:
                    continue
                file_path = pathlib.Path(root) / name
                rel_file = rel_root / name
                try:
                    stat = file_path.stat()
                except FileNotFoundError:
                    continue
                entries.append({
                    "path": rel_file.as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()
        return entries

    if use_progress and tqdm is not None:
        entries = _build_index()
    else:
        entries = await asyncio.to_thread(_build_index)
    index_path.write_text(json.dumps(entries, indent=2))
    return index_path


def report_to_dict(report: AnnotatedReport) -> dict[str, object]:
    summary = getattr(report.report, "summary", "")
    try:
        body_dict = asdict(report.report)
    except TypeError:
        body_dict = {
            "summary": summary,
            "vulns": getattr(report.report, "vulns", []),
        }
    return {
        "function": report.member.name.decode(errors="replace"),
        "file": report.member.file.path,
        "summary": summary,
        "vulns": list(report.vulns),
        "body": body_dict,
    }


def compute_quantile_threshold(scores: list[float], q: float) -> float:
    if not scores:
        return 0.0
    if not 0 < q < 1:
        raise ValueError("q must be in (0,1)")
    scores_sorted = sorted(scores)
    idx = max(0, math.ceil(q * len(scores_sorted)) - 1)
    return scores_sorted[idx]


def flatten_stage2_reports(
    annotated_reports: list[AnnotatedReport], stage_label: str, model: str | None
) -> list[Stage2Record]:
    flattened: list[Stage2Record] = []
    for report in annotated_reports:
        body_range = getattr(report.member, "body", report.member.range)
        text_snippet = report.member.file[body_range].decode(errors="replace")
        base = report_to_dict(report)
        function_name = cast(str, base["function"])
        file_path = cast(str, base["file"])
        summary_text = cast(str, base["summary"])
        report_body = cast(dict[str, object], base["body"])
        for vuln in report.vulns:
            record: Stage2Record = {
                "stage": stage_label,
                "model": model,
                "function": function_name,
                "file": file_path,
                "description": vuln,
                "summary": summary_text,
                "code_snippet": text_snippet,
                "report": report_body,
            }
            flattened.append(record)
    return flattened


async def stage2_single(project: AnalysisProject, model: str) -> list[Stage2Record]:
    _, annotated = await analyze_project(project, model=model)
    return flatten_stage2_reports(annotated, "single", model)


async def stage2_multi(project: AnalysisProject, model: str) -> list[Stage2Record]:
    _, annotated = await analyze_project_multifunc(project, model=model)
    return flatten_stage2_reports(annotated, "multi", model)


async def stage3_scoring(records: Iterable[Stage2Record], project_name: str, batch_size: int) -> list[Stage3Score]:
    scored: list[Stage3Score] = []
    for idx, record in enumerate(records, start=1):
        vuln_text = record["description"]
        code_text = record["code_snippet"]
        batch = await LikelyVulnClassifier.batch_classify(batch_size, project_name, vuln_text, code_text)
        scored.append({
            "index": idx,
            "function": record["function"],
            "file": record["file"],
            "description": record["description"],
            "stage": record["stage"],
            "avg_likely": batch.avg("likely"),
            "max_likely": batch.max("likely"),
            "std_likely": batch.std("likely"),
        })
    return scored


async def stage3_trace(
    records: Iterable[Stage2Record],
    project_name: str,
    batch_size: int,
    model: str,
    score_threshold: float,
    root_dir: pathlib.Path,
    analysis_project: AnalysisProject,
    include_non_new: bool,
) -> list[Stage3Trace]:
    trace: list[Stage3Trace] = []
    candidates: list[AnalyzedVuln] = []
    harness = FakeHarness(name="default_harness", source="unknown")
    searcher = FakeSearcher(analysis_project)
    fake_crs = FakePovCRS(
        project=FakeProject(name=project_name, info=FakeProjectInfo()),
        harnesses=[harness],
        root_dir=root_dir,
        analysis_project=analysis_project,
        searcher=searcher,
    )
    for record in records:
        vuln_text = record["description"]
        code_text = record["code_snippet"]
        analyzed = AnalyzedVuln(
            function=record["function"],
            file=record["file"],
            description=record["description"],
            conditions=record["summary"],
        )
        fake_crs.current_file = pathlib.Path(record["file"]) if record.get("file") else None
        batch = await LikelyVulnClassifier.batch_classify(batch_size, project_name, vuln_text, code_text)
        score = batch.max("likely")
        if score < score_threshold:
            continue
        analysis_result, analysis_error = await run_vuln_analyzer(fake_crs, record)
        analysis_payload = _analysis_summary(analysis_result, analysis_error)
        analyzed_for_dedupe = analyzed
        if analysis_result is not None and analysis_result.positive is not None:
            analyzed_for_dedupe = analysis_result.positive

        dedup_result = await DedupClassifier(project_name, analyzed_for_dedupe, candidates).classify()
        key, prob = dedup_result.best()
        dedupe_choice = "NEW" if key == "NEW" else str(key)
        dedupe_confidence = prob
        candidates.append(analyzed_for_dedupe)
        if dedupe_choice != "NEW" and not include_non_new:
            continue
        triage_note = "(disabled) TriageAgent requires a real POV + harness run"
        pov_note = await run_pov_agent(fake_crs, analyzed_for_dedupe, model_idx=0)
        pov_note = _merge_pov_note(pov_note, analysis_payload)
        trace.append({
            "function": record["function"],
            "file": record["file"],
            "description": record["description"],
            "likely_confidence": score,
            "likely_above_threshold": score > score_threshold,
            "dedupe_choice": dedupe_choice,
            "dedupe_confidence": dedupe_confidence,
            "triage_note": triage_note,
            "pov_note": pov_note,
            "vuln_analysis": analysis_payload,
        })
    return trace


async def run(args: argparse.Namespace) -> None:
    target = pathlib.Path(args.directory).resolve()
    if not target.is_dir():
        raise SystemExit(f"{target} is not a directory")

    use_progress = not args.no_progress

    env_path = pathlib.Path(args.env_file) if args.env_file else pathlib.Path(".env")
    loaded_env = {}
    if env_path.exists():
        loaded_env = load_env_from_file(env_path, silent=True)
        if loaded_env:
            print(f"loaded {len(loaded_env)} vars from {env_path}")

    if args.no_probs:
        os.environ["CLASSIFIER_ALLOW_STRUCTURED"] = "1"

    if not args.skip_testflight:
        await run_testflight(args.model)

    project_name = args.project_name or target.name

    analysis_source = target
    ingest_meta: dict[str, object] = {"enabled": not args.skip_ingest}
    if not args.skip_ingest:
        cache_root = args.cache_dir
        cache_root.mkdir(parents=True, exist_ok=True)
        project_hash = args.project_hash or await compute_project_hash(target)
        exclude_set = set(args.ingest_exclude or DEFAULT_INGEST_EXCLUDES)
        index_path = await stage0_index(target, cache_root, project_hash, exclude_set, use_progress)
        ingest_meta.update({
            "hash": project_hash,
            "index": str(index_path),
            "cache_dir": str(cache_root / project_hash / target.name),
            "excludes": sorted(exclude_set),
        })
        print(f"stage0: indexed {index_path}")
    else:
        ingest_meta["skipped"] = True

    project = await build_analysis_project(analysis_source, use_progress)
    results: list[Stage2Record] = []

    if args.mode in ("single", "both"):
        if use_progress and tqdm is not None:
            with tqdm(total=1, desc="Stage 2 single", unit="phase", leave=True) as bar:
                results.extend(await stage2_single(project, args.model))
                bar.update(1)
        else:
            results.extend(await stage2_single(project, args.model))
    if args.mode in ("multi", "both"):
        if use_progress and tqdm is not None:
            with tqdm(total=1, desc="Stage 2 multi", unit="phase", leave=True) as bar:
                results.extend(await stage2_multi(project, args.model_multi))
                bar.update(1)
        else:
            results.extend(await stage2_multi(project, args.model_multi))

    scoring_iter = results
    if use_progress and tqdm is not None:
        scoring_iter = tqdm(results, desc="Stage 3 scoring", total=len(results), unit="item", leave=True)
    scored = await stage3_scoring(scoring_iter, project_name, args.batch)
    score_values = [entry["max_likely"] for entry in scored]
    quantile_threshold = compute_quantile_threshold(score_values, args.score_quantile)
    if args.use_quantile:
        effective_threshold = max(args.score_threshold, quantile_threshold)
    else:
        effective_threshold = args.score_threshold
    thresholded_records = [
        record
        for record, score in zip(results, scored, strict=False)
        if score["max_likely"] >= effective_threshold
    ]
    trace_iter = thresholded_records
    if use_progress and tqdm is not None:
        trace_iter = tqdm(thresholded_records, desc="Stage 3 trace", total=len(thresholded_records), unit="item", leave=True)
    traced = await stage3_trace(
        trace_iter,
        project_name,
        args.batch,
        args.model,
        effective_threshold,
        analysis_source,
        project,
        args.include_non_new,
    )

    output = {
        "project": project_name,
        "directory": target.as_posix(),
        "stage2_findings": results,
        "stage3_scores": scored,
        "stage3_trace": traced,
        "stage3_threshold": effective_threshold,
        "stage3_threshold_floor": args.score_threshold,
        "stage3_threshold_quantile": args.score_quantile,
        "stage3_threshold_quantile_value": quantile_threshold,
        "stage3_threshold_use_quantile": args.use_quantile,
        "stage3_trace_include_non_new": args.include_non_new,
        "stage0_ingest": ingest_meta,
        "analysis_source": analysis_source.as_posix(),
    }

    args.output.write_text(json.dumps(output, indent=2))
    new_traces = sum(1 for entry in traced if entry.get("dedupe_choice") == "NEW")
    print(f"wrote {args.output} with {len(results)} findings ({len(scored)} scored, {new_traces} traced new)")


DEFAULT_MODEL = os.getenv("STAGE023_MODEL") or os.getenv("MODEL") or "anthropic/claude-3.5-2024-12-17"
DEFAULT_MODEL_MULTI = os.getenv("STAGE023_MODEL_MULTI") or os.getenv("MODEL_MULTI") or DEFAULT_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only stage 2/3 analysis on a local repo")
    _ = parser.add_argument("directory", type=pathlib.Path, help="path to the repository to analyze")
    _ = parser.add_argument("--mode", choices=("single", "multi", "both"), default="single")
    _ = parser.add_argument("--model", default=DEFAULT_MODEL, help="model for the single-function prompt")
    _ = parser.add_argument("--model-multi", default=DEFAULT_MODEL_MULTI, help="model for multi-function prompt")
    _ = parser.add_argument("--batch", type=int, default=2, help="LikelyVulnClassifier batch size for scoring")
    _ = parser.add_argument("--score-threshold", type=float, default=0.1, help="minimum likely score to run deeper stage 3 trace")
    _ = parser.add_argument("--score-quantile", type=float, default=0.8, help="quantile threshold to match CRS gating")
    _ = parser.add_argument("--use-quantile", action="store_true", help="apply quantile threshold in addition to fixed score threshold")
    _ = parser.add_argument("--include-non-new", action="store_true", help="include non-NEW dedupe entries in stage3_trace")
    _ = parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    _ = parser.add_argument("--skip-testflight", action="store_true", help="skip LLM connectivity check")
    _ = parser.add_argument("--project-name", help="label to give the analyzed project (defaults to directory name)")
    _ = parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("stage023-output.json"))
    _ = parser.add_argument("--skip-ingest", action="store_true", help="skip stage 0 ingestion and run analysis straight against the working tree")
    _ = parser.add_argument("--no-probs", action="store_true", help="Bypass logprob requirement from gpt models using structured output.")
    _ = parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE_ROOT, help="where to store stage 0 ingest tarballs")
    _ = parser.add_argument("--project-hash", help="override the project hash used for ingestion cache directories")
    _ = parser.add_argument("--ingest-exclude", nargs="*", default=sorted(DEFAULT_INGEST_EXCLUDES), help="top-level directories or files to skip when creating the ingest tarball")
    _ = parser.add_argument("--env-file", type=pathlib.Path, help="path to dotenv file (defaults to .env)")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
