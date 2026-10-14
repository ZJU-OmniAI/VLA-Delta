import logging
import math
import signal
import threading
import time
from collections import deque

import torch


class WandbTrainingLogger:
    def __init__(
        self,
        accelerator,
        *,
        project,
        run_name,
        run_id,
        resume,
        log_dir,
        group=None,
        tags=None,
        config=None,
        failure_policy="best_effort",
        pending_capacity=256,
        retry_base_steps=1,
        retry_max_steps=128,
        finish_max_attempts=2,
        finish_timeout_seconds=15.0,
        batch_metric_reductions=False,
        resume_safe_steps=False,
        resume_attempt=0,
    ):
        self.accelerator = accelerator
        self.enabled = bool(project and project.strip())
        self.run = None
        self.batch_metric_reductions = bool(batch_metric_reductions)
        self.resume_safe_steps = bool(resume_safe_steps)
        self.resume_attempt = int(resume_attempt)
        for name, value in (
            ("pending_capacity", pending_capacity),
            ("retry_base_steps", retry_base_steps),
            ("retry_max_steps", retry_max_steps),
            ("finish_max_attempts", finish_max_attempts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"W&B {name} must be a positive integer")
        if retry_max_steps < retry_base_steps:
            raise ValueError("W&B retry_max_steps must be at least retry_base_steps")
        if (
            isinstance(finish_timeout_seconds, bool)
            or not isinstance(finish_timeout_seconds, (int, float))
            or not math.isfinite(finish_timeout_seconds)
            or finish_timeout_seconds <= 0
        ):
            raise ValueError("W&B finish_timeout_seconds must be a finite positive number")
        self.pending_capacity = pending_capacity
        self.retry_base_steps = retry_base_steps
        self.retry_max_steps = retry_max_steps
        self.finish_max_attempts = finish_max_attempts
        self.finish_timeout_seconds = float(finish_timeout_seconds)
        self._pending = deque()
        self.dropped_payload_count = 0
        self.consecutive_failures = 0
        self.next_retry_step = 0
        self.remote_recovery_count = 0
        self.remote_disable_count = 0
        self.remote_available = False
        self.finish_timed_out = False
        self.remote_abandoned = False
        self.remote_finished = False
        failure_policy = failure_policy or ("required" if self.enabled else "best_effort")
        if failure_policy not in {"best_effort", "required"}:
            raise ValueError("W&B failure_policy must be best_effort or required")
        self.failure_policy = failure_policy
        init_error = None

        if self.enabled and accelerator.is_main_process:
            try:
                import wandb

                self.run = wandb.init(
                    project=project,
                    name=run_name or None,
                    id=run_id or None,
                    resume=resume,
                    dir=log_dir or None,
                    group=group or None,
                    tags=tags,
                    config=config,
                    mode="online",
                    save_code=False,
                    settings=wandb.Settings(disable_code=True, disable_git=True),
                )
                if self.resume_safe_steps:
                    self.run.define_metric("train/global_step")
                    self.run.define_metric("train/*", step_metric="train/global_step")
            except Exception as error:
                init_error = error

        if self.enabled:
            local_success = not accelerator.is_main_process or self.run is not None
            if getattr(accelerator, "num_processes", 1) > 1:
                status = torch.tensor(
                    int(local_success), dtype=torch.int64, device=accelerator.device
                )
                local_success = bool(
                    accelerator.reduce(status, reduction="min").item()
                )
            if not local_success:
                if failure_policy == "required":
                    raise RuntimeError("W&B initialization failed on the main rank") from init_error
                self.enabled = False
                self.remote_disable_count += 1
                if accelerator.is_main_process:
                    logging.getLogger(__name__).warning(
                        "W&B initialization failed; continuing with authoritative local logs: %s",
                        init_error,
                    )
        self.remote_available = self.run is not None

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "wandb_pending_payloads": len(self._pending),
            "wandb_dropped_payloads": self.dropped_payload_count,
            "wandb_consecutive_failures": self.consecutive_failures,
            "wandb_next_retry_step": (
                self.next_retry_step if self.consecutive_failures else -1
            ),
            "wandb_remote_enabled": int(self.enabled),
            "wandb_remote_available": int(self.remote_available),
            "wandb_remote_disable_count": self.remote_disable_count,
            "wandb_remote_recovery_count": self.remote_recovery_count,
            "wandb_finish_timed_out": int(self.finish_timed_out),
            "wandb_finish_timeout_seconds": self.finish_timeout_seconds,
            "wandb_remote_abandoned": int(self.remote_abandoned),
            "wandb_remote_finished": int(self.remote_finished),
        }

    def _enqueue(self, payload) -> None:
        if len(self._pending) >= self.pending_capacity:
            self._pending.popleft()
            self.dropped_payload_count += 1
        self._pending.append(payload)

    def _register_failure(self, step: int, error: Exception) -> None:
        self.consecutive_failures += 1
        exponent = min(self.consecutive_failures - 1, 62)
        delay = min(self.retry_base_steps * (2**exponent), self.retry_max_steps)
        self.next_retry_step = int(step) + delay
        self.remote_available = False
        logging.getLogger(__name__).warning(
            "W&B logging failed; pending=%d dropped=%d failures=%d "
            "next_retry_step=%d while local logs remain authoritative: %s",
            len(self._pending),
            self.dropped_payload_count,
            self.consecutive_failures,
            self.next_retry_step,
            error,
        )

    def _flush_pending(self, *, step: int, max_attempts: int | None = None) -> None:
        attempts = 0
        recovered = self.consecutive_failures > 0
        while self._pending and (max_attempts is None or attempts < max_attempts):
            metrics, pending_step = self._pending[0]
            attempts += 1
            try:
                self.run.log(metrics, **({} if self.resume_safe_steps else {"step": pending_step}))
            except Exception as error:
                if self.failure_policy == "required":
                    raise RuntimeError("required W&B logging failed") from error
                self._register_failure(step, error)
                return
            self._pending.popleft()
        if recovered:
            self.remote_recovery_count += 1
            logging.getLogger(__name__).warning(
                "W&B remote logging recovered; pending=%d dropped=%d",
                len(self._pending),
                self.dropped_payload_count,
            )
        self.consecutive_failures = 0
        self.next_retry_step = int(step)
        self.remote_available = True

    def log(self, *, step, mean_metrics, scalar_metrics):
        if not self.enabled:
            return self.diagnostics()

        reduced_metrics = {}
        numeric_names, numeric_values = [], []
        for name, value in mean_metrics.items():
            if isinstance(value, (str, bool)):
                reduced_metrics[name] = value
                continue
            if self.batch_metric_reductions:
                numeric_names.append(name)
                numeric_values.append(torch.as_tensor(value, device=self.accelerator.device,
                    dtype=torch.float32).detach().reshape(()))
                continue
            reduced_value = self.accelerator.reduce(
                torch.as_tensor(value, device=self.accelerator.device, dtype=torch.float32).detach(),
                reduction="mean",
            )
            if self.accelerator.is_main_process:
                reduced_metrics[name] = reduced_value.item()

        if numeric_values:
            reduced_values = self.accelerator.reduce(torch.stack(numeric_values), reduction="mean")
            if self.accelerator.is_main_process:
                reduced_metrics.update(zip(numeric_names, reduced_values.detach().cpu().tolist()))

        if self.run is not None:
            payload_metrics = {**reduced_metrics, **scalar_metrics}
            if self.resume_safe_steps:
                payload_metrics.update({"train/global_step": int(step), "resume_attempt": self.resume_attempt})
            payload = (payload_metrics, step)
            self._enqueue(payload)
            if not self.consecutive_failures or int(step) >= self.next_retry_step:
                self._flush_pending(step=int(step))
        return self.diagnostics()

    @staticmethod
    def _call_before_deadline(function, deadline: float):
        """Interrupt one main-thread remote call at the monotonic deadline."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout", None
        if threading.current_thread() is not threading.main_thread():
            return "timeout", None

        class _DeadlineExpired(BaseException):
            pass

        def expire(signum, frame):
            del signum, frame
            raise _DeadlineExpired()

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        started = time.monotonic()
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            return "success", function()
        except _DeadlineExpired:
            return "timeout", None
        except Exception as error:
            return "error", error
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                elapsed = time.monotonic() - started
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(previous_timer[0] - elapsed, 1e-9),
                    previous_timer[1],
                )

    def _abandon_remote_after_timeout(self) -> None:
        self.finish_timed_out = True
        self.remote_abandoned = True
        self.remote_available = False
        self.enabled = False
        self.remote_disable_count += 1
        self.run = None
        logging.getLogger(__name__).warning(
            "W&B best-effort finish timed out after %.3f seconds; remote run was "
            "abandoned with pending=%d dropped=%d failures=%d",
            self.finish_timeout_seconds,
            len(self._pending),
            self.dropped_payload_count,
            self.consecutive_failures,
        )

    def _finish_best_effort(self, exit_code: int) -> None:
        deadline = time.monotonic() + self.finish_timeout_seconds
        attempts = 0
        while self._pending and attempts < self.finish_max_attempts:
            metrics, pending_step = self._pending[0]
            status, value = self._call_before_deadline(
                lambda metrics=metrics, pending_step=pending_step: self.run.log(
                    metrics, **({} if self.resume_safe_steps else {"step": pending_step})
                ),
                deadline,
            )
            if status == "timeout":
                self._abandon_remote_after_timeout()
                return
            attempts += 1
            if status == "error":
                self._register_failure(max(self.next_retry_step, 0), value)
                break
            self._pending.popleft()

        active_run = self.run
        status, value = self._call_before_deadline(
            lambda: active_run.finish(exit_code=exit_code), deadline
        )
        if status == "timeout":
            self._abandon_remote_after_timeout()
            return
        self.run = None
        self.enabled = False
        self.remote_available = False
        if status == "error":
            self.remote_abandoned = True
            self.remote_disable_count += 1
            logging.getLogger(__name__).warning(
                "W&B finish failed after local training state was saved: %s", value
            )
            return
        self.remote_finished = True

    def finish(self, exit_code=0):
        if self.run is not None:
            if self.failure_policy == "required":
                if self._pending:
                    self._flush_pending(
                        step=max(self.next_retry_step, 0),
                        max_attempts=self.finish_max_attempts,
                    )
                try:
                    self.run.finish(exit_code=exit_code)
                except Exception as error:
                    raise RuntimeError("required W&B finish failed") from error
                self.remote_finished = True
                self.run = None
                self.remote_available = False
                self.enabled = False
            else:
                self._finish_best_effort(exit_code)
        return self.diagnostics()
