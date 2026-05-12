#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: executor.py
#   Author: xyy15926
#   Created: 2026-05-10 16:17:59
#   Updated: 2026-05-12 20:13:22
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import threading
import asyncio
import inspect
import contextvars
import functools
from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from flagbear.sched import task, protocols
    reload(task)
    reload(protocols)

from flagbear.sched.protocols import(
    _current_context,
    TaskState, 
    TaskResult,
    TaskFuture,
    RetryPolicy,
    ExecutionPolicy,
    Context,
)

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


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
        self.futures = {}

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
                empty_ctx = contextvars.Context()
                try:
                    empty_ctx.run(loop.run_forever)
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

    def run(
        self,
        ctx: Context,
        *task_futures: TaskFuture,
    ) -> list[TaskResult]:
        """Submit task and wait until the task complete.

        Params:
        ----------------------------
        ctx: Context to get and set the TaskResult.
        task_future: TaskFutures.

        Return:
        ----------------------------
        TaskResult or a list of TaskResult correspondant to the `task_future`
        passed.
        """
        self.submit(ctx, *task_futures)
        return self.gather(*task_futures)

    def submit(
        self,
        ctx: Context,
        *task_futures: TaskFuture,
    ):
        """Submit task to inner event loop.

        Task submited will be scheduled and executed background.

        Params:
        ----------------------------
        ctx: Context to get and set the TaskResult.
        task_future: TaskFutures.
        """
        # set_trace()
        for tfut in task_futures:
            coro = self.execute_with_context(tfut, ctx)
            future = self.run_coro(coro)
            self.futures[tfut.id_] = future

    def gather(
        self,
        *task_futures: TaskFuture,
    ) -> list[TaskResult]:
        """Wait for the Future.

        Params:
        ----------------------------
        futures: Concurrent futures.

        Return:
        ----------------------------
        list of TaskResult
        """
        results = []
        for tfut in task_futures:
            if tfut.id_ not in self.futures:
                raise ValueError(f"Result of task {tfut.name} "
                                 f"has been gather.")
            future = self.futures.pop(tfut.id_)
            results.append(future.result())
        return results

    # TODO: move the sync task wrapper out of the coroutine.
    def run_coro(self, coro: "coroutine") -> Future:
        """Run coroutine in inner event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
        return future

    async def execute_with_context(
        self,
        task_future: TaskFuture,
        ctx: Context,
    ) -> TaskResult:
        """Execute the TaskFuture async.

        Context will be use to get resolve the dependencies and arguments
        of the task future.
        1. TaskFutures depended on will be resolved and executed first.
        2. Arguments will be then resolved to get the actual value from the
          TaskFutures passed as arguments.
        2.1. If any TaskFutures in arguments is unready, RuntimeError will be
          raised, since all dependencies should be executed.
        2.2. If any TaskFutures in arguments failed, current TaskFuture will
          be skipped.

        Params:
        ----------------------------
        task_future: TaskFuture.
        ctx: Context to get and set the TaskResult.

        Return:
        ----------------------------
        TaskReult
        """
        task_name = task_future.name

        # Resolve arguments.
        (resolved_args, resolved_kwargs,
         unready_tasks, failed_tasks) = task_future.resolve_args(ctx)
        if len(failed_tasks) > 0:
            task_names = ",".join([f.name for f in failed_tasks.keys()])
            error = RuntimeError(
                f"Skip to execute task {task_name} because precedent "
                f"tasks: {task_names} failed."
            )
            result = TaskResult(
                TaskState.SKIPPED,
                None,
                error,
                datetime.now(),
                None,
                0,
            )
            # ctx.set_result(task_future, result)
            return result

        # All tasks should be ready or failed, since all dependent tasks
        # has been awaited.
        if len(unready_tasks) > 0:
            task_names = ",".join([f.name for f in unready_tasks])
            raise RuntimeError(
                f"Failed to execute task {task_name} because precedent "
                f"tasks: {task_names} can't be ready."
            )

        logger.info(f"Wait for task {task_name} to start.")
        result = await self.execute_resolved(
            task_future,
            resolved_args,
            resolved_kwargs,
        )
        logger.info(f"Task {task_name} ends.")

        return result

    async def execute_resolved(
        self,
        task_future: TaskFuture,
        resolved_args,
        resolved_kwargs,
    ) -> TaskResult:
        """Execute task future with resolved arguments.

        Retry policy will be considered here to determine if to retry.

        Params:
        --------------------------
        task_future: TaskFuture.
        resolved_args: Positional arugments with actual value.
        resolved_kwargs: Keywords arugments with actual value.

        Return:
        --------------------------
        TaskResult
        """
        func = task_future.func
        is_async = task_future.is_async
        task_name = task_future.name

        retry_policy = task_future.retry_policy or RetryPolicy()
        execution_policy = task_future.execution_policy or ExecutionPolicy()
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
                    logger.error(f"Task {task_name} failed: {error}.")
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
                    logger.error(f"Task {task_name} failed: {error}.")
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
        execution_policy: Optional[ExecutionPolicy] = None,
        is_async: Optional[bool] = None,
    ) -> Any:
        """Execute function.

        1. For async function, coroutine will be awaited directly.
        2. For sync funtion, thread or process will be gotten from inner
          threadpool or processpool to execute and then wrapped in a 
          async Future to be scheduled by the event-loop.

        Params:
        --------------------------
        func: Callable.
        resolved_args: Positional arugments with actual value.
        resolved_kwargs: Keywords arugments with actual value.
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

        # Clear `_current_context` for the async task.
        # As async task should never #TODO
        if is_async or not with_context:
            _current_context.set(None)

        # Await async task function.
        if is_async:
            if timeout is not None:
                value = await asyncio.wait_for(
                    func(*resolved_args, **resolved_kwargs),
                    timeout = timeout,
                )
            else:
                value = await func(*resolved_args, **resolved_kwargs)
        # Run sync task function in threadpool or process pool, but
        # scheduled by asyncio by `asyncio.wrap_future`.
        # 1. `asyncio.to_thread`：contextvars.copy_context and then
        #   `asyncio.get_running_loop().run_in_executor` the sync
        #   function(task) within given executor or default executor.
        # 2. `loop.run_in_executor` will `asyncio.wrap_future` the future
        #   return by `executor.submit` the sync function(task).
        else:
            loop = self.event_loop
            pool = (self.thread_executor if not use_process
                    else self.process_executor)
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
    ctx = contextvars.copy_context()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return ctx.run(func, *args, **kwargs)

    return wrapper


async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)
