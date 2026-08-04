#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: executor.py
#   Author: xyy15926
#   Created: 2026-05-10 16:17:59
#   Updated: 2026-05-20 22:25:27
#   Description:
# ---------------------------------------------------------

# %%
import asyncio
import contextvars
import functools
import inspect
import logging
import threading
from collections.abc import Coroutine
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from typing import Any

# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload

    from flagbear.sched import protocols, task
    from flagbear.slp import cache
    reload(cache)
    reload(task)
    reload(protocols)

from flagbear.sched.protocols import (
    ExecutionPolicy,
    RetryPolicy,
    Task,
    TaskResult,
    TaskState,
    _current_context,
)
from flagbear.slp.cache import Cache

logger = logging.getLogger(__name__)


# %%
class LocalExecutor:
    """Executor use asyncio.event_loop to schedule tasks.

    Attrs:
    ---------------------------------
    max_threads: The maximum threads to execute task in parallel.
    max_processes: The maximum processes to execute task in parallal.
    _thread_pool: Inner thread pool to execute sync task in sub-thread.
    _process_pool: Inner process pool to execute sync task in sub-process.
    _event_loop: Inner event loop to schedule tasks.

    Ref:
    ---------------------------------
    - https://zhuanlan.zhihu.com/p/1982801155877798503
    - https://runebook.dev/zh/docs/python/library/asyncio-future/asyncio.wrap_future
    """

    def __init__(
        self,
        max_threads: int = 4,
        max_processes: int = 0,
    ):
        self.max_threads = max_threads
        self.max_processes = max_processes
        self._thread_pool = None
        self._process_pool = None
        self._event_loop = None

    @property
    def thread_executor(self):
        """Init ThreadPoolExecutor lazily."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(self.max_threads)
        return self._thread_pool

    @property
    def process_executor(self):
        """Init ProcessPoolExecutor lazily."""
        if self.max_processes == 0:
            return self._thread_pool
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(self.max_processes)
        return self._process_pool

    @property
    def event_loop(self):
        """Inner async event loop."""
        if self._event_loop is None:
            loop = asyncio.new_event_loop()
            self._event_loop = loop

            def loop_thread(loop):
                asyncio.set_event_loop(loop)
                # empty_ctx = contextvars.Context()
                try:
                    # empty_ctx.run(loop.run_forever)
                    loop.run_forever()
                finally:
                    loop.close()

            # 1. Run event loop in another thread to ensure the loop will
            #   always run and tasks submited will be scheduled immidiately.
            # 2. `daemon = True` will kill the loop-thread when the main
            #   thread die so to avoid hang the process up.
            t = threading.Thread(
                target = loop_thread,
                args=(loop,),
                daemon = True,
            )
            t.start()

        return self._event_loop

    def shutdown_event_loop(self):
        """Shutdown event loop."""
        # Cancel the pending tasks in event loop.
        async def wait_for_loop():
            tasks = [t for t in asyncio.all_tasks()
                     if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            # Wait for the tasks to be cancelled.
            await asyncio.gather(*tasks, return_exceptions = True)

        loop = self._event_loop
        if loop.is_running():
            # Wait for the task cancelling task.
            asyncio.run_coroutine_threadsafe(
                wait_for_loop(),
                loop,
            ).result()
            # Then stop the loop.
            loop.call_soon_threadsafe(loop.stop)

    def shutdown(self):
        """Shutdown event-loop, thread-pool and process-pool."""
        if self._event_loop is not None:
            self.shutdown_event_loop()
            logger.info("Shut down event loop.")
        if self._thread_pool is not None:
            self._thread_pool.shutdown()
            logger.info("Shut down thread pool.")
        if self._process_pool is not None:
            self._process_pool.shutdown()
            logger.info("Shut down process pool.")

    def submit(
        self,
        task_results: Cache,
        *tasks: Task | tuple[Task, callable],
    ) -> list[Future] | None:
        """Submit tasks."""
        # Wrap nofity as `asyncio.Task` callback.
        def on_done(
            fut: asyncio.Future,
            task: Task,
            notify: callable | None = None,
        ):
            try:
                _result = fut.result()
            # Catch the `asyncio.CancelledError` which may be caused by
            # `shutdown`.
            except asyncio.CancelledError as e:
                error = e
                result = TaskResult(
                    TaskState.CANCELLED,
                    None,
                    None,
                    error,
                    datetime.now(),
                    None,
                    0,
                )
                task_result_cache_policy = result.cache_policy()
                task_results.set(task.id_, result, task_result_cache_policy)
            except Exception:
                logger.exception("Task execution failed.")
            finally:
                if notify:
                    notify()

        # Wrap task into a function to create a `asyncio.Task` that will be
        # passed to and executed in event-loop.
        def task_wrapper(task, notify):
            coro = self.execute_with_cache(task, task_results)
            atask = asyncio.create_task(coro)
            callback = functools.partial(on_done, task = task, notify = notify)
            atask.add_done_callback(callback)

        futures = []
        for task in tasks:
            # If task and nofity are passed together, nofity will be added
            # to `asyncio.Task` to as callback directly instead of creating
            # a `concurrent.futures.Future` and return.
            if isinstance(task, tuple):
                task, notify = task
                self.event_loop.call_soon_threadsafe(
                    task_wrapper,
                    task,
                    notify,
                )
            else:
                coro = self.execute_with_cache(task, task_results)
                future = self.run_coro(coro)
                futures.append(future)

        # Return `concurrent.futures.Future` to if necessary.
        if len(futures) > 0:
            return futures

    def run_coro(self, coro: Coroutine) -> Future:
        """Run coroutine in inner event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
        return future

    async def execute_with_cache(
        self,
        task: Task,
        task_results: Cache,
    ) -> TaskResult:
        """Execute the Task async.

        Context will be use to get resolve the dependencies and arguments
        of the task future.
        1. Tasks depended on will be resolved and executed first.
        2. Arguments will be then resolved to get the actual value from the
          Tasks passed as arguments.
        2.1. If any Tasks in arguments is unready, RuntimeError will be
          raised, since all dependencies should be executed.
        2.2. If any Tasks in arguments failed, current Task will
          be skipped.

        Params:
        ----------------------------
        task: Task.
        ctx: Context to get and set the TaskResult.

        Return:
        ----------------------------
        TaskReult
        """
        task_name = task.name
        # Clear `_current_context` so that nested `Tasks` won't get
        # `_current_context` and will perform just as a normal function.
        _current_context.set(None)
        value_cache_policy = task.cache_policy

        # Resolve arguments.
        (resolved_args, resolved_kwargs,
         unready_tasks, failed_tasks) = task.resolve_args(task_results)
        if len(failed_tasks) > 0:
            task_names = ",".join([f.name for f in failed_tasks])
            error = RuntimeError(
                f"Skip to execute task {task_name} because precedent "
                f"tasks: {task_names} failed."
            )
            result = TaskResult(
                TaskState.SKIPPED,
                None,
                None,
                error,
                datetime.now(),
                None,
                0,
            )
            task_result_cache_policy = result.cache_policy(value_cache_policy)
            task_results.set(task.id_, result, task_result_cache_policy)
            return result

        # All tasks should be ready or failed, since all dependent tasks
        # has been awaited.
        if len(unready_tasks) > 0:
            task_names = ",".join([f.name for f in unready_tasks])
            raise RuntimeError(
                f"Failed to execute task {task_name} because precedent "
                f"tasks: {task_names} can't be ready."
            )

        cache_key = task.cache_key
        func_result = task_results.get(cache_key)
        if func_result is not None:
            result = TaskResult(
                TaskState.CACHED,
                cache_key,
                func_result,
                None,
                datetime.now(),
                datetime.now(),
                0,
            )
            task_result_cache_policy = result.cache_policy(value_cache_policy)
            task_results.set(task.id_, result, task_result_cache_policy)
            logger.info(f"Use cache for task {task_name}.")
            return result

        logger.info(f"Wait for task {task_name} to start.")
        result = await self.execute_resolved(
            task,
            resolved_args,
            resolved_kwargs,
        )
        if result.is_successful():
            task_results.set(cache_key, result.value, value_cache_policy)
        task_result_cache_policy = result.cache_policy(value_cache_policy)
        task_results.set(task.id_, result, task_result_cache_policy)
        logger.info(f"Task {task_name} ends.")

        return result

    async def execute_resolved(
        self,
        task: Task,
        resolved_args,
        resolved_kwargs,
    ) -> TaskResult:
        """Execute task future with resolved arguments.

        Retry policy will be considered here to determine if to retry.

        Params:
        --------------------------
        task: Task.
        resolved_args: Positional arguments with actual value.
        resolved_kwargs: Keywords arguments with actual value.

        Return:
        --------------------------
        TaskResult
        """
        func = task.func
        is_async = task.is_async
        task_name = task.name

        retry_policy = task.retry_policy or RetryPolicy()
        execution_policy = task.execution_policy or ExecutionPolicy()
        timeout = execution_policy.timeout

        # Excecute task.
        state = TaskState.RUNNING
        start_time = datetime.now()
        end_time = None
        value = None
        attempt = 0
        error = None

        # set_trace()
        while True:
            attempt += 1
            logger.info(f"Task {task_name} starts at {attempt} attempt.")
            try:
                value = await self.execute_func_once(
                    func,
                    resolved_args,
                    resolved_kwargs,
                    execution_policy,
                    is_async,
                )
                state = TaskState.SUCCESS
                value = value
                break
            except asyncio.TimeoutError:
                # Repack error.
                error = TimeoutError(
                    f"Task {task_name} timed out after {timeout}s."
                )
                state = TaskState.FAILED
                if not retry_policy.should_retry(error, attempt):
                    logger.exception(f"Task {task_name} failed: {error}.")
                    # logger.exception(error)
                    break

                # Retry.
                state = TaskState.RETRYING
                delay = retry_policy.get_delay(attempt)
                logger.info(
                    f"Task {task_name} time out after {timeout}s, retrying "
                    f"in {delay}s."
                )
                await asyncio.sleep(delay)
            except Exception as e:
                # set_trace()
                # Variable `e` in `except as` will be deleted even after
                # overriding by inner block.
                # So a new variable `error` should be used.
                error = e
                state = TaskState.FAILED
                if not retry_policy.should_retry(error, attempt):
                    logger.exception(f"Task {task_name} failed: {error}.")
                    # logger.exception(error)
                    break

                # Retry.
                state = TaskState.RETRYING
                delay = retry_policy.get_delay(attempt)
                logger.info(
                    f"Task {task_name} failed: {error}, retrying in {delay}s."
                )
                await asyncio.sleep(delay)

        end_time = datetime.now()
        result = TaskResult(
            state,
            task.cache_key,
            value,
            error,
            start_time,
            end_time,
            attempt,
        )
        return result

    async def execute_func_once(
        self,
        func,
        resolved_args: list[Any],
        resolved_kwargs: dict[str, Any],
        execution_policy: ExecutionPolicy | None = None,
        is_async: bool | None = None,
    ) -> Any:
        """Execute function.

        1. For async function, coroutine will be awaited directly.
        2. For sync funtion, thread or process will be gotten from inner
          threadpool or processpool to execute and then wrapped in a
          async Future to be scheduled by the event-loop.

        Params:
        --------------------------
        func: Callable.
        resolved_args: Positional arguments with actual value.
        resolved_kwargs: Keywords arguments with actual value.
        execution_policy: Execution policty.
        is_async: If function is a async function.

        Return:
        --------------------------
        Return value of `func`.
        """
        is_async = (inspect.iscoroutinefunction(func)
                    if is_async is None
                    else is_async)
        execution_policy = execution_policy or ExecutionPolicy()
        timeout = execution_policy.timeout
        use_process = execution_policy.use_process
        with_context = execution_policy.with_context

        # Await async task function.
        if is_async:
            if timeout is not None:
                value = await asyncio.wait_for(
                    func(*resolved_args, **resolved_kwargs),
                    timeout = timeout,
                )
            else:
                value = await func(*resolved_args, **resolved_kwargs)
        else:
            pool = (self.thread_executor if not use_process
                    else self.process_executor)
            loop = asyncio.get_running_loop()
            # Copy context.
            if with_context:
                func = with_copied_context(func)
            future = pool.submit(func, *resolved_args, **resolved_kwargs)
            async_future = asyncio.wrap_future(future, loop = loop)
            if timeout is not None:
                value = await asyncio.wait_for(
                    async_future,
                    timeout = timeout,
                )
            else:
                value = await async_future

        return value


# %%
def with_copied_context(func):
    """Copy context for function execution."""
    ctx = contextvars.copy_context()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return ctx.run(func, *args, **kwargs)

    return wrapper
