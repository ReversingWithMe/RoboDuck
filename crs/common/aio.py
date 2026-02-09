from collections.abc import Buffer
from pathlib import PurePath
from typing import Any, AsyncGenerator, AsyncIterator, Callable, IO, Iterator, Literal, Self, TypeAlias, cast
import asyncio
import contextlib
import io
import os
import pathlib
import shutil
import stat
import tempfile

from crs.common.shield import SoloTaskGroup, shield_and_wait, finalize

StrPath = os.PathLike[str] | str
BytesPath = os.PathLike[bytes] | bytes
AnyPath = StrPath | BytesPath

@contextlib.asynccontextmanager
async def open(
    path: AnyPath,
    mode: str = "r",
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
) -> AsyncIterator[IO[Any]]:
    f: IO[Any] | None = None
    async def cleanup():
        if f is not None:
            await asyncio.to_thread(f.close)

    async with finalize(cleanup()):
        f = await asyncio.to_thread(io.open, file=path, mode=mode, encoding=encoding, errors=errors, newline=newline)
        yield f

@contextlib.asynccontextmanager
async def iterator[T](it: Iterator[T]) -> AsyncIterator[AsyncIterator[T]]:
    queue = asyncio.Queue[T]()
    loop = asyncio.get_running_loop()
    def source():
        try:
            for obj in it:
                fut = asyncio.run_coroutine_threadsafe(queue.put(obj), loop)
                fut.result()
            _ = loop.call_soon_threadsafe(queue.shutdown)
        except asyncio.QueueShutDown:
            pass

    async def aiter() -> AsyncIterator[T]: # noqa: ASYNC900 # we're in a context manager
        try:
            while True:
                yield await queue.get()
        except asyncio.QueueShutDown:
            pass

    async with SoloTaskGroup() as tg:
        try:
            _ = tg.create_task(asyncio.to_thread(source), name="aio.iterator")
            yield aiter()
        finally:
            queue.shutdown(immediate=True)

# from typeshed
OpenTextModeUpdating: TypeAlias = Literal[
    "r+", "+r", "rt+", "r+t", "+rt", "tr+", "t+r", "+tr", "w+", "+w", "wt+", "w+t",
    "+wt", "tw+", "t+w", "+tw", "a+", "+a", "at+", "a+t", "+at", "ta+", "t+a", "+ta",
    "x+", "+x", "xt+", "x+t", "+xt", "tx+", "t+x", "+tx",
]
OpenTextModeWriting: TypeAlias = Literal["w", "wt", "tw", "a", "at", "ta", "x", "xt", "tx"]
OpenTextModeReading: TypeAlias = Literal["r", "rt", "tr", "U", "rU", "Ur", "rtU", "rUt", "Urt", "trU", "tUr", "Utr"]
OpenTextMode: TypeAlias = OpenTextModeUpdating | OpenTextModeWriting | OpenTextModeReading
OpenBinaryModeUpdating: TypeAlias = Literal[
    "rb+", "r+b", "+rb", "br+", "b+r", "+br", "wb+", "w+b", "+wb", "bw+", "b+w", "+bw",
    "ab+", "a+b", "+ab", "ba+", "b+a", "+ba", "xb+", "x+b", "+xb", "bx+", "b+x", "+bx",
]
OpenBinaryModeWriting: TypeAlias = Literal["wb", "bw", "ab", "ba", "xb", "bx"]
OpenBinaryModeReading: TypeAlias = Literal["rb", "br", "rbU", "rUb", "Urb", "brU", "bUr", "Ubr"]
OpenBinaryMode: TypeAlias = OpenBinaryModeUpdating | OpenBinaryModeReading | OpenBinaryModeWriting

async def async_map[A, B](fn: Callable[[A], B], it: AsyncIterator[A]) -> AsyncGenerator[B, None]: # noqa: ASYNC900 # this is only used from a context manager
    async for item in it:
        yield fn(item)

class Path(PurePath):
    async def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        fn = os.stat if follow_symlinks else os.lstat
        return await asyncio.to_thread(fn, self)

    async def chmod(self, mode: int, *, follow_symlinks: bool = True) -> None:
        return await asyncio.to_thread(os.chmod, self, mode, follow_symlinks=follow_symlinks)

    # NOTE: pathlib may suppress some errors on Path.is_dir and friends
    async def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        st = await self.stat(follow_symlinks=follow_symlinks)
        return stat.S_ISDIR(st.st_mode)

    async def is_file(self, *, follow_symlinks: bool = True) -> bool:
        st = await self.stat(follow_symlinks=follow_symlinks)
        return stat.S_ISREG(st.st_mode)

    async def read_text(self, encoding: str | None = None, errors: str | None = None, newline: str | None = None) -> str:
        path = pathlib.Path(self)
        return await asyncio.to_thread(path.read_text, encoding=encoding, errors=errors, newline=newline)

    @contextlib.asynccontextmanager
    async def glob(self, pattern: str, *, case_sensitive: bool | None = None, recurse_symlinks: bool = False) -> AsyncIterator[AsyncIterator[Self]]:
        path = pathlib.Path(self)
        it = await asyncio.to_thread(path.glob, pattern=pattern, case_sensitive=case_sensitive, recurse_symlinks=recurse_symlinks)
        async with iterator(it) as async_it:
            yield async_map(self.__class__, async_it)

    @contextlib.asynccontextmanager
    async def rglob(self, pattern: str, *, case_sensitive: bool | None = None, recurse_symlinks: bool = False) -> AsyncIterator[AsyncIterator[Self]]:
        path = pathlib.Path(self)
        it = await asyncio.to_thread(path.rglob, pattern=pattern, case_sensitive=case_sensitive, recurse_symlinks=recurse_symlinks)
        async with iterator(it) as async_it:
            yield async_map(self.__class__, async_it)

    async def exists(self, *, follow_symlinks: bool = True) -> bool:
        fn = os.path.exists if follow_symlinks else os.path.lexists
        return await asyncio.to_thread(fn, self)

    async def is_symlink(self) -> bool:
        st = await self.stat(follow_symlinks=False)
        return stat.S_ISLNK(st.st_mode)

    async def is_socket(self) -> bool:
        st = await self.stat(follow_symlinks=False)
        return stat.S_ISSOCK(st.st_mode)

    async def is_fifo(self) -> bool:
        st = await self.stat(follow_symlinks=False)
        return stat.S_ISFIFO(st.st_mode)

    async def is_block_device(self) -> bool:
        st = await self.stat(follow_symlinks=False)
        return stat.S_ISBLK(st.st_mode)

    async def is_char_device(self) -> bool:
        st = await self.stat(follow_symlinks=False)
        return stat.S_ISCHR(st.st_mode)

    @contextlib.asynccontextmanager
    async def iterdir(self) -> AsyncIterator[AsyncIterator[Self]]:
        path = pathlib.Path(self)
        it = await asyncio.to_thread(path.iterdir)
        async with iterator(it) as async_it:
            yield async_map(self.__class__, async_it)

    async def lchmod(self, mode: int) -> None:
        await self.chmod(mode, follow_symlinks=False)

    async def lstat(self) -> os.stat_result:
        return await asyncio.to_thread(os.lstat, self)

    async def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        await asyncio.to_thread(pathlib.Path(self).mkdir, mode=mode, parents=parents, exist_ok=exist_ok)

    async def readlink(self) -> Self:
        return self.__class__(await asyncio.to_thread(os.readlink, self))

    async def rename(self, target: AnyPath) -> None:
        await asyncio.to_thread(os.rename, self, target)

    async def replace(self, target: AnyPath) -> None:
        await asyncio.to_thread(os.replace, self, target)

    async def resolve(self, strict: bool=False) -> Self:
        return self.__class__(await asyncio.to_thread(os.path.realpath, self, strict=strict))

    async def rmdir(self, strict: bool=False) -> None:
        await asyncio.to_thread(os.rmdir, self)

    async def symlink_to(self, target: AnyPath, target_is_directory: bool = False) -> None:
        await asyncio.to_thread(os.symlink, self, target)

    async def touch(self, mode: int=0o666, exist_ok: bool=True) -> None:
        path = pathlib.Path(self)
        await asyncio.to_thread(path.touch, mode=mode, exist_ok=exist_ok)

    async def unlink(self, missing_ok: bool=False) -> None:
        try:
            await asyncio.to_thread(os.unlink, self)
        except FileNotFoundError:
            if not missing_ok:
                raise

    @classmethod
    def home(cls) -> Self:
        return cls(pathlib.Path.home())

    async def sync(self) -> pathlib.Path:
        return pathlib.Path(self)

    async def absolute(self) -> Self:
        return self.__class__(await asyncio.to_thread(os.path.abspath, self))

    async def expanduser(self) -> Self:
        return self.__class__(await asyncio.to_thread(os.path.expanduser, self))

    async def read_bytes(self) -> bytes:
        path = pathlib.Path(self)
        return await asyncio.to_thread(path.read_bytes)

    async def samefile(self, other: AnyPath) -> bool:
        return await asyncio.to_thread(os.path.samefile, self, other)

    async def write_bytes(self, data: Buffer) -> int:
        path = pathlib.Path(self)
        return await asyncio.to_thread(path.write_bytes, data)

    async def write_text(
        self, data: str, encoding: str | None = None, errors: str | None = None, newline: str | None = None
    ) -> int:
        path = pathlib.Path(self)
        return await asyncio.to_thread(path.write_text, data, encoding=encoding, errors=errors, newline=newline)

    @contextlib.asynccontextmanager
    async def walk(
        self, top_down: bool = False, on_error: Callable[[OSError], object] | None = None, follow_symlinks: bool = False,
    ) -> AsyncIterator[AsyncIterator[tuple[Self, list[str], list[str]]]]:
        proxy = pathlib.Path(self)
        it = await asyncio.to_thread(proxy.walk, top_down=top_down, on_error=on_error, follow_symlinks=follow_symlinks)
        async with iterator(it) as async_it:
            async def wrap_it(): # noqa: ASYNC900 # we're in a context manager
                async for root, dirs, names in async_it:
                    yield self.__class__(root), dirs, names
            yield wrap_it()

class PathIO(IO[bytes]):
    path: Path

@contextlib.asynccontextmanager
async def tmpdir(suffix: str | None = None, prefix: str | None = None, dir: StrPath | None = None) -> AsyncIterator[Path]:
    path = await shield_and_wait(asyncio.to_thread(tempfile.mkdtemp, suffix=suffix, prefix=prefix, dir=dir))

    async def cleanup():
        await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)

    async with finalize(cleanup()):
        yield Path(path)

@contextlib.asynccontextmanager
async def tmpfile(
    *,
    mode: str = "w+b",
    suffix: str | None = None,
    prefix: str | None = None,
    dir: StrPath | None = None,
    delete: bool = True
) -> AsyncIterator[PathIO]:
    (fd, path) = await shield_and_wait(asyncio.to_thread(tempfile.mkstemp, prefix=prefix, suffix=suffix, dir=dir))
    f: IO[bytes] | None = None
    path = Path(path)

    async def cleanup():
        if f is not None:
            await asyncio.to_thread(f.close)
        if delete:
            try:
                await path.unlink(missing_ok=True)
            except OSError:
                pass

    async with finalize(cleanup()):
        f = await asyncio.to_thread(os.fdopen, fd, mode=mode)
        f = cast(PathIO, f)
        # fdopen uses the fd for the name, fix it
        if isinstance(f, io.TextIOWrapper):
            f.buffer.raw.name = path.as_posix() # type: ignore
        else:
            f.raw.name = path.as_posix() # type: ignore

        f.path = path
        yield f

async def batch_unlink(*paths: str, missing_ok: bool=False) -> None:
    def unlink():
        for path in paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                if not missing_ok:
                    raise
    await asyncio.to_thread(unlink)
