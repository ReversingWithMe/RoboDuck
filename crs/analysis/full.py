from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, TypedDict, Optional, Iterable, cast
import argparse
import asyncio
import os
import re
import time
import traceback

from litellm.utils import get_max_tokens
import litellm
litellm.suppress_debug_info = True

from tqdm import tqdm
import orjson
import yaml

from . import c_tree_sitter
from . import java_tree_sitter
from .data import AnalysisProject, SourceFile, SourceMember, SourceFunction, AnnotatedReport
from .parser import parse_body, parse_body_multifunc

from crs_rust import logger
from crs.common.prompts import prompt_manager

Function = TypedDict(
    'Function', {
        'name': str,
        'fullname': str
    }
)

QueryResult = TypedDict(
    'QueryResult', {
        'path': str,
        'function': Function,
        'model': str,
        'messages': list[dict[str, str]],
        'usage': dict[str, Any],
        'cost': float,
        'elapsed': float
    }
)

RawQueryResult = TypedDict(
    'RawQueryResult', {
        'model': str,
        'messages': list[dict[str, str]],
        'usage': dict[str, Any],
        'cost': float,
        'elapsed': float
    }
)

def unzip_pairs[X, Y](it: Iterable[tuple[X, Y]]) -> tuple[Iterable[X], Iterable[Y]]:
    return tuple(zip(*it)) # type: ignore

async def _crs_completion(model: str, messages: list[dict[str, Any]]):
    from crs.common.llm_api import priority_completion
    from crs.common.workdb import cur_job_priority
    from crs.common.types import Priority
    return await priority_completion(cur_job_priority.get(Priority.HIGH), model=model, messages=messages)

def _cache_key(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(["{role}: {content}".format(**msg) for msg in messages[:-1]])

TRIVIAL_RE = re.compile(rb"^[^\[\(\*@>]*$")

def filter_members(members: list[SourceMember], min_lines: int | None = None) -> list[SourceMember]:
    result: list[SourceMember] = []
    seen: set[bytes] = set()
    for m in members:
        if not isinstance(m, SourceFunction):
            result.append(m)
            continue
        body = m.file[m.body]
        if not body.strip(b"{} \t\r\n"):
            continue
        a, b = m.file.range_to_lines(m.body)
        line_count = b - a
        if min_lines:
            if line_count < min_lines:
                continue
        if len(body) > 50_000:
            continue
        if body in seen:
            continue
        seen.add(body)
        if m.name in (b"LLVMFuzzerTestOneInput", "fuzzerTestOneInput"):
            continue
        if line_count <= 4 and TRIVIAL_RE.match(body):
            continue
        result.append(m)
    return result

async def query(member: SourceMember, model: str, system: str, user: str, cache: Optional[dict[str, Any]] = None) -> QueryResult:
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if cache is not None:
            if (key := _cache_key(messages)) in cache:
                return cache[key]
            else:
                raise RuntimeError("cache miss")
        start = time.perf_counter()
        response = (await _crs_completion(model=model, messages=messages)).unwrap()
        cost = response.cost
        elapsed = time.perf_counter() - start
        message = cast(litellm.Choices, response.choices[0]).message
        messages.append({
            "role": message.role,
            "content": cast(str, message.content),
        })
        m: Function = {
            "name": member.name.decode(errors="replace"),
            "fullname": member.fullname.decode(errors="replace")
        }
        usage: dict[str, Any] = response.usage if isinstance(response.usage, dict) else response.usage.model_dump() # type: ignore
        result: QueryResult = {
            "path": member.file.path,
            "function": m,
            "model": model,
            "messages": messages,
            "usage": usage,
            "cost": cost,
            "elapsed": elapsed,
        }
        return result
    except Exception:
        traceback.print_exc()
        raise

async def query_raw(model: str, system: str, user: str, cache: Optional[dict[str, Any]] = None) -> RawQueryResult:
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if cache is not None:
            if (key := _cache_key(messages)) in cache:
                return cache[key]
            else:
                raise RuntimeError("cache miss")
        start = time.perf_counter()
        response = (await _crs_completion(model=model, messages=messages)).unwrap()
        cost = response.cost
        elapsed = time.perf_counter() - start
        message = cast(litellm.Choices, response.choices[0]).message
        messages.append({
            "role": message.role,
            "content": cast(str, message.content),
        })
        usage: dict[str, Any] = response.usage if isinstance(response.usage, dict) else response.usage.model_dump() # type: ignore
        result: RawQueryResult = {
            "model": model,
            "messages": messages,
            "usage": usage,
            "cost": cost,
            "elapsed": elapsed,
        }
        return result
    except Exception:
        traceback.print_exc()
        raise

# TODO: multiple triage views: from the perspective of each harness, versus from the raw source
# TODO: want compile commands + CU filtering!
# TODO: there will be extra files in docker - the repo tar / diff they give us will have a more constrained list of files
# TODO: cache various kinds of analysis, on both a file:line and function basis?

def load_cache(cache_path: str, model: str) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    with open(cache_path, "r") as f:
        for line in f:
            j = orjson.loads(line)
            if j["model"] != model:
                continue
            key = _cache_key(j["messages"])
            cache[key] = j
    return cache

async def analyze_project(project: AnalysisProject, progress: bool = False, model: Optional[str] = None, cache_path: Optional[str] = None) -> tuple[list[QueryResult], list[AnnotatedReport]]:
    model = model or "gpt-4o-2024-08-06"
    model_prompts = prompt_manager.model(model)
    prompt_map = {
        "c": model_prompts.bind("FullModeSingleC", kwargs={}),
        "java": model_prompts.bind("FullModeSingleJava", kwargs={}),
    }
    unknown_prompts = model_prompts.bind("FullModeSingleUnknown", kwargs={})

    if cache_path is not None:
        cache = await asyncio.to_thread(load_cache, cache_path, model)
    else:
        cache = None

    sem = asyncio.Semaphore(10_000)
    async def analyze_one(member: SourceMember) -> Optional[tuple[QueryResult, AnnotatedReport]]:
        sf = member.file
        lang = sf.path.rsplit(".", 1)[-1]
        prompts = prompt_map.get(lang, unknown_prompts)

        # WARNING: we are mutating prompts.kwargs to avoid calling prompts.bind() again
        # this is _only_ safe if there's no checkpoint before we fetch the .user / .system attrs
        try:
            sf = member.file
            source_range = sf.expand_range_to_lines(member.range)
            prompts.kwargs.update({
                "fullname": member.fullname.decode(),
                "source": sf[source_range].decode(),
            })
            system = prompts.system
            user = prompts.user
        except UnicodeDecodeError as e:
            _ = pbar.update(1)
            logger.warning("Exception analyzing {member} with {model}: {exc}", member=member.fullname, model=model, exc=repr(e))
            return None

        async with sem:
            result = None
            try:
                result = await query(member, model=model, system=system, user=user, cache=cache)
                report = await asyncio.to_thread(parse_body, result["messages"][-1]["content"])
            except Exception as e:
                _ = pbar.update(1)
                if not result:
                    logger.exception("Exception analyzing {member} with {model}", member=member.fullname, model=model, exc=e)
                else:
                    logger.warning("Exception analyzing {member} with {model}: {exc}", member=member.fullname, model=model, exc=repr(e))
                return None
            vulns: list[str] = []
            for vuln in report.vulns:
                desc = await asyncio.to_thread(yaml.dump, vuln, default_flow_style=False)
                if lang == "java":
                    sanitizer = vuln.get("sanitizer", "")
                    if sanitizer_desc := getattr(prompts.custom, f"jazzer_{sanitizer}", None):
                        desc = f"{desc}\n\nUsing the {sanitizer!r} Jazzer sanitizer:\n\n{sanitizer_desc}"
                vulns.append(desc)
            _ = pbar.update(1)
            return result, AnnotatedReport(member, report, vulns)

    decls = await asyncio.to_thread(filter_members, project.decls)
    with tqdm(total=len(decls), desc="analysis", disable=not progress) as pbar:
        output = await asyncio.gather(*[analyze_one(member) for member in decls])
    query_results, reports = unzip_pairs((x for x in output if x is not None))
    try:
        llm_results, reports = list(query_results), list(reports)
    except ValueError:
        llm_results, reports = [], []
    return llm_results, reports

@dataclass(slots=True)
class LanguageGroup:
    name: str
    chunks: list[str]
    chunk: list[str]
    chunksize: int
    totalsize: int

async def analyze_project_multifunc(project: AnalysisProject, progress: bool = False, model: Optional[str] = None, cache_path: Optional[str] = None) -> tuple[list[RawQueryResult], list[AnnotatedReport]]:
    model = model or "gemini/gemini-2.5-pro"
    model_prompts = prompt_manager.model(model)
    prompt_map = {
        "c": model_prompts.bind("FullModeMultiC", kwargs={}),
        "java": model_prompts.bind("FullModeMultiJava", kwargs={}),
    }
    unknown_prompts = model_prompts.bind("FullModeMultiUnknown", kwargs={})

    if cache_path is not None:
        cache = await asyncio.to_thread(load_cache, cache_path, model)
    else:
        cache = None

    try:
        max_tokens = get_max_tokens(model) or 65535
    except Exception:
        max_tokens = 65535
    sem = asyncio.Semaphore(50)
    all_files = {sf.path: f"// path: {path}\n{sf.source.decode(errors='replace')}\n" for path, sf in sorted(project.files.items())}

    # TODO: token counting? or split the chunk if we get a context length error?
    maxsize = int(max_tokens * 4)
    splitsize = int(maxsize * 2)

    token_to_sf: dict[bytes, list[SourceFile]] = defaultdict(list)
    token_count: dict[bytes, int] = defaultdict(int)
    path_to_tokens: dict[str, set[bytes]] = {}

    # compute token counts
    for sf in project.files.values():
        tokens = set(re.findall(rb"\w{4,}", sf.source))
        for token in tokens:
            token_count[token] += 1
            token_to_sf[token].append(sf)
        path_to_tokens[sf.path] = tokens

    groups: dict[str, LanguageGroup] = {}

    remaining: set[str] = set(project.files.keys())
    active_tokens: set[bytes] = set()
    while remaining:
        next_sf: SourceFile | None = None

        # do a best-token match first
        if active_tokens:
            token_sort = sorted(active_tokens, key=lambda t: (token_count[t], -len(t)))
            for token in token_sort:
                for candidate in token_to_sf[token]:
                    if candidate.path in remaining:
                        next_sf = candidate
                        remaining.remove(candidate.path)
                        active_tokens |= {t for t in path_to_tokens[candidate.path] if token_count[t] > 1}
                        break
                else:
                    continue
                break

        if next_sf is None:
            # fall back to any old file
            next_sf = project.files[sorted(remaining)[0]]
            active_tokens |= {t for t in path_to_tokens[next_sf.path] if token_count[t] > 1}
            remaining.remove(next_sf.path)

        path = next_sf.path
        data = all_files[path]
        ext = path.rsplit(".", 1)[-1]
        if (group := groups.get(ext)) is None:
            group = LanguageGroup(
                name=ext,
                chunks=[],
                chunk=[],
                chunksize=0,
                totalsize=0,
            )
            groups[ext] = group

        if len(data) > maxsize:
            # TODO: warn? split? do this earlier when we create the file text mappings
            continue

        if group.chunksize + len(data) > splitsize:
            group.chunks.append("\n".join(group.chunk))
            group.chunk.clear()
            group.chunksize = 0
            active_tokens.clear()

        group.chunk.append(data)
        group.chunksize += len(data)
        group.totalsize += len(data)

    for group in groups.values():
        if group.chunk:
            group.chunks.append("\n".join(group.chunk))
            group.chunk.clear()
            group.chunksize = 0

    async def analyze_one(lang: str, chunk: str) -> Optional[tuple[RawQueryResult, list[AnnotatedReport]]]:
        prompts = prompt_map.get(lang, unknown_prompts)

        # WARNING: we are mutating prompts.kwargs to avoid calling prompts.bind() again
        # this is _only_ safe if there's no checkpoint before we fetch the .user / .system attrs
        prompts.kwargs.update({
            "source": chunk,
        })
        system = prompts.system
        user = prompts.user

        async with sem:
            result = None
            try:
                result = await query_raw(model, system, user, cache=cache)
                _ = pbar.update(1)
                report = await asyncio.to_thread(parse_body_multifunc, result["messages"][-1]["content"])
            except Exception as e:
                if not result:
                    logger.exception("Exception analyzing multi with {model}", model=model, exc=e)
                else:
                    logger.warning("Exception analyzing multi with {model}: {exc}", model=model, exc=repr(e))
                return None

            vuln_map: dict[SourceMember, AnnotatedReport] = {}
            for vuln in report.vulns:
                for f in vuln.functions:
                    try:
                        fname = f["name"]
                        fpath = f["path"]
                        reason = f.get("reason")
                        source = f.get("source")
                    except KeyError:
                        continue
                    # try to match to a SourceMember
                    # TODO: if we CAN'T match a SourceMember, report without one? this mitigates source parsing bugs
                    if (member := project.name_to_decl.get(fname.encode())) is None:
                        continue
                    if os.path.basename(fpath) != os.path.basename(member.file.path):
                        continue
                    if (annotated := vuln_map.get(member)) is None:
                        annotated = AnnotatedReport(member, report, [])
                        vuln_map[member] = annotated
                    desc_map: dict[str, str] = {
                        "name": vuln.name,
                    }
                    if reason is not None:
                        desc_map["reason"] = reason
                    if source is not None:
                        desc_map["source"] = source
                    desc = "\n".join(f"{k}: {v}" for k, v in desc_map.items())
                    if lang == "java" and (sanitizer_desc := getattr(prompts.custom, f"jazzer_{vuln.name}", None)):
                        desc = f"{desc}\n\nUsing the {vuln.name!r} Jazzer sanitizer:\n\n{sanitizer_desc}"
                    annotated.vulns.append(desc)

            return result, list(vuln_map.values())

    with tqdm(total=sum(len(group.chunks) for group in groups.values()), desc="multifunc", disable=not progress) as pbar:
        output: list[Optional[tuple[RawQueryResult, list[AnnotatedReport]]]] = []
        for lang, group in groups.items():
            output += [await analyze_one(lang, group.chunks[0])]
            output += await asyncio.gather(*[analyze_one(lang, chunk) for chunk in group.chunks[1:]])

    query_results, report_lists = unzip_pairs((x for x in output if x is not None))
    try:
        llm_results, reports = list(query_results), [r for l in report_lists for r in l]
    except ValueError:
        llm_results, reports = [], []
    return llm_results, reports

# main() impl for testing outside the CRS:

def find_files(path: Path, exts: tuple[str]) -> list[Path]:
    if path.is_file():
        return [path]
    results: list[Path] = []
    for root, _dirs, names in path.walk():
        for name in names:
            if name.endswith(exts):
                results.append(root / name)
    return results

async def main():
    from crs import config # noqa: F401, force logger setup # pyright: ignore [reportUnusedImport]
    logger.set_level("WARNING")

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("src", nargs="+")
    _ = parser.add_argument("--name", help="output name", required=True)
    _ = parser.add_argument("--out", help="output directory", required=True)
    _ = parser.add_argument("--multifunc", help="use multifunc analysis", action="store_true")
    _ = parser.add_argument("--model", help="use specific model")
    _ = parser.add_argument("--cache", help="cached json from a previous run")
    args = parser.parse_args()

    project = AnalysisProject()
    for root in args.src:
        root = Path(root)
        for path in tqdm(find_files(root, (".c", ".java")), desc="load files"): # type: ignore
            # TODO: store root somewhere? different project per root? have a different "project root"?
            source = path.read_bytes()
            path = path.relative_to(root) if path != root else path
            strpath = path.as_posix()
            assert strpath not in project.files
            sf = SourceFile(strpath, source)
            project.files[strpath] = sf

            if path.name.endswith(".c"):
                decls = c_tree_sitter.parse(sf)
                project.decls += decls
            elif path.name.endswith(".java"):
                decls = java_tree_sitter.parse(sf)
                project.decls += decls
            else:
                ...
    project.build_lut()

    if args.multifunc:
        llm_results, reports = await analyze_project_multifunc(project, progress=True, model=args.model, cache_path=args.cache)
    else:
        llm_results, reports = await analyze_project(project, progress=True, model=args.model, cache_path=args.cache)

    dt = datetime.now(tz=UTC).isoformat()
    name = f"{args.name}_{dt}"
    if args.model:
        name = f"{args.model}_{name}".replace("/", "_")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name)

    with open(f"{path}-results.jsonl", "w") as f: # noqa: ASYNC230, not in main crs path
        for row in llm_results:
            jline = json.dumps(row)
            _ = f.write(f"{jline}\n") # noqa: ASYNC232, not in main crs path

    with open(f"{path}-report.jsonl", "w") as f: # noqa: ASYNC230, not in main crs path
        for obj in reports:
            objd = asdict(obj)
            objd.pop("member", None)
            objd = {"path": obj.member.file.path, "fullname": obj.member.fullname.decode(errors="replace"), **objd}
            jline = json.dumps(objd)
            _ = f.write(f"{jline}\n") # noqa: ASYNC232, not in main crs path

if __name__ == "__main__":
    asyncio.run(main())
