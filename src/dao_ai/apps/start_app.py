"""
Start script for running the chat UI frontend alongside the dao-ai agent backend.

Clones the Databricks e2e-chatbot-app-next template (if not present), builds it,
and starts both the frontend (port 3000) and backend (port 8000) concurrently.
The MLflow AgentServer's enable_chat_proxy proxies UI requests to the frontend.

When enable_chat_proxy is False in the dao-ai config, skips the frontend entirely
and only starts the backend agent server.

Usage:
    python -m dao_ai.apps.start_app
    python -m dao_ai.apps.start_app --no-ui
    python -m dao_ai.apps.start_app --port 8000
"""

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

BACKEND_READY_PATTERNS = [
    r"Uvicorn running on",
    r"Application startup complete",
    r"Started server process",
]
FRONTEND_READY_PATTERNS = [r"Server is running on http://localhost"]

CHAT_APP_DIR = "e2e-chatbot-app-next"
REPO_URL_HTTPS = "https://github.com/databricks/app-templates.git"
REPO_URL_SSH = "git@github.com:databricks/app-templates.git"


def _check_port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except OSError:
        return False


class ProcessManager:
    def __init__(self, port: int = 8000, no_ui: bool = False) -> None:
        self.backend_process: subprocess.Popen | None = None
        self.frontend_process: subprocess.Popen | None = None
        self.backend_ready = False
        self.frontend_ready = False
        self.failed = threading.Event()
        self.backend_log = None
        self.frontend_log = None
        self.port = port
        self.no_ui = no_ui

    def check_ports(self) -> None:
        errors: list[str] = []
        if not _check_port_available(self.port):
            errors.append(
                f"Port {self.port} (backend) is already in use.\n"
                f"  To free it: lsof -ti :{self.port} | xargs kill -9"
            )

        if not self.no_ui:
            frontend_port = int(
                os.environ.get("CHAT_APP_PORT", os.environ.get("PORT", "3000"))
            )
            if self.port == frontend_port:
                print(
                    f"ERROR: Backend and frontend are both configured to use port {self.port}."
                )
                print(
                    "  Set CHAT_APP_PORT in the environment to a different port (e.g., CHAT_APP_PORT=3000)."
                )
                sys.exit(1)

            if not _check_port_available(frontend_port):
                errors.append(
                    f"Port {frontend_port} (frontend) is already in use.\n"
                    f"  To free it: lsof -ti :{frontend_port} | xargs kill -9\n"
                    f"  Or set a different port: CHAT_APP_PORT=<port>"
                )

        if errors:
            print("ERROR: Port(s) already in use:\n")
            for error in errors:
                print(f"  {error}\n")
            sys.exit(1)

    def monitor_process(
        self,
        process: subprocess.Popen,
        name: str,
        log_file,
        patterns: list[str],
    ) -> None:
        is_ready = False
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                line = line.rstrip()
                log_file.write(line + "\n")
                print(f"[{name}] {line}")

                if not is_ready and any(
                    re.search(p, line, re.IGNORECASE) for p in patterns
                ):
                    is_ready = True
                    if name == "backend":
                        self.backend_ready = True
                    else:
                        self.frontend_ready = True
                    print(f"  {name.capitalize()} is ready!")

                    if self.no_ui and self.backend_ready:
                        print(f"\n{'=' * 50}")
                        print("  Backend is ready! (running without UI)")
                        print(f"  API available at http://localhost:{self.port}")
                        print(f"{'=' * 50}\n")
                    elif self.backend_ready and self.frontend_ready:
                        print(f"\n{'=' * 50}")
                        print("  Both frontend and backend are ready!")
                        print(f"  Open the frontend at http://localhost:{self.port}")
                        print(f"{'=' * 50}\n")

            process.wait()
            if process.returncode != 0:
                self.failed.set()

        except Exception as e:
            print(f"Error monitoring {name}: {e}")
            self.failed.set()

    def clone_frontend_if_needed(self) -> bool:
        if Path(CHAT_APP_DIR).exists():
            return True

        print(f"Cloning {CHAT_APP_DIR}...")
        for url in [REPO_URL_HTTPS, REPO_URL_SSH]:
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--filter=blob:none",
                        "--sparse",
                        url,
                        "temp-app-templates",
                    ],
                    check=True,
                    capture_output=True,
                )
                break
            except subprocess.CalledProcessError:
                continue
        else:
            print("ERROR: Failed to clone app-templates repository.")
            print(
                "Manually download from: "
                "https://download-directory.github.io/?url="
                "https://github.com/databricks/app-templates/tree/main/e2e-chatbot-app-next"
            )
            return False

        subprocess.run(
            ["git", "sparse-checkout", "set", CHAT_APP_DIR],
            cwd="temp-app-templates",
            check=True,
        )
        Path(f"temp-app-templates/{CHAT_APP_DIR}").rename(CHAT_APP_DIR)
        shutil.rmtree("temp-app-templates", ignore_errors=True)
        return True

    def start_process(
        self,
        cmd: list[str],
        name: str,
        log_file,
        patterns: list[str],
        cwd: str | None = None,
    ) -> subprocess.Popen:
        print(f"Starting {name}...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )

        thread = threading.Thread(
            target=self.monitor_process,
            args=(process, name, log_file, patterns),
            daemon=True,
        )
        thread.start()
        return process

    def print_logs(self, log_path: str) -> None:
        print(f"\nLast 50 lines of {log_path}:")
        print("-" * 40)
        try:
            lines = Path(log_path).read_text().splitlines()
            print("\n".join(lines[-50:]))
        except FileNotFoundError:
            print(f"(no {log_path} found)")
        print("-" * 40)

    def cleanup(self) -> None:
        print(f"\n{'=' * 42}")
        print("Shutting down...")
        print(f"{'=' * 42}")

        for proc in [self.backend_process, self.frontend_process]:
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, Exception):
                    proc.kill()

        if self.backend_log:
            self.backend_log.close()
        if self.frontend_log:
            self.frontend_log.close()

    def run(self, backend_args: list[str] | None = None) -> int:
        load_dotenv(dotenv_path=Path(".env.local"), override=True)
        if not os.environ.get("DATABRICKS_APP_NAME"):
            self.check_ports()

        if not self.no_ui:
            if not self.clone_frontend_if_needed():
                print(
                    "WARNING: Failed to clone frontend. Continuing with backend only."
                )
                self.no_ui = True
            else:
                os.environ["API_PROXY"] = f"http://localhost:{self.port}/invocations"

        self.backend_log = open("backend.log", "w", buffering=1)
        if not self.no_ui:
            self.frontend_log = open("frontend.log", "w", buffering=1)

        try:
            backend_cmd = [
                sys.executable,
                "-m",
                "dao_ai.apps.server",
                "--port",
                str(self.port),
            ]
            if backend_args:
                backend_cmd.extend(backend_args)

            self.backend_process = self.start_process(
                backend_cmd, "backend", self.backend_log, BACKEND_READY_PATTERNS
            )

            if not self.no_ui:
                frontend_dir = Path(CHAT_APP_DIR)
                npm_bin = shutil.which("npm")
                if not npm_bin:
                    print("ERROR: npm not found. Install Node.js to use the chat UI.")
                    print("Continuing with backend only.")
                    self.no_ui = True
                else:
                    for cmd_str, desc in [
                        ("npm install", "install"),
                        ("npm run build", "build"),
                    ]:
                        print(f"Running npm {desc}...")
                        result = subprocess.run(
                            cmd_str.split(),
                            cwd=frontend_dir,
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode != 0:
                            print(f"npm {desc} failed: {result.stderr}")
                            print("Continuing with backend only.")
                            self.no_ui = True
                            break

                if not self.no_ui:
                    self.frontend_process = self.start_process(
                        ["npm", "run", "start"],
                        "frontend",
                        self.frontend_log,
                        FRONTEND_READY_PATTERNS,
                        cwd=str(frontend_dir),
                    )
                    print(
                        f"\nMonitoring processes "
                        f"(Backend PID: {self.backend_process.pid}, "
                        f"Frontend PID: {self.frontend_process.pid})\n"
                    )
                else:
                    print(
                        f"\nMonitoring backend process (PID: {self.backend_process.pid})\n"
                    )
            else:
                print(
                    f"\nMonitoring backend process (PID: {self.backend_process.pid})\n"
                )

            while not self.failed.is_set():
                time.sleep(0.1)
                if self.backend_process.poll() is not None:
                    self.failed.set()
                    break
                if (
                    not self.no_ui
                    and self.frontend_process
                    and self.frontend_process.poll() is not None
                ):
                    self.failed.set()
                    break

            failed_name = "backend"
            failed_proc = self.backend_process
            if (
                not self.no_ui
                and self.frontend_process
                and self.backend_process.poll() is None
            ):
                failed_name = "frontend"
                failed_proc = self.frontend_process
            exit_code = failed_proc.returncode if failed_proc else 1

            print(
                f"\n{'=' * 42}\n"
                f"ERROR: {failed_name} process exited with code {exit_code}\n"
                f"{'=' * 42}"
            )
            self.print_logs("backend.log")
            if not self.no_ui:
                self.print_logs("frontend.log")
            return exit_code

        except KeyboardInterrupt:
            print("\nInterrupted")
            return 0

        finally:
            self.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start dao-ai agent with optional chat UI frontend",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run backend only, skip frontend UI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the backend server (default: 8000)",
    )
    args, backend_args = parser.parse_known_args()
    sys.exit(ProcessManager(port=args.port, no_ui=args.no_ui).run(backend_args))


if __name__ == "__main__":
    main()
