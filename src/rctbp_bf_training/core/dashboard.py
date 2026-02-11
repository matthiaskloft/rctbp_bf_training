"""
Optuna Dashboard Launcher.

CLI and programmatic interface for launching the Optuna Dashboard
to monitor optimization studies in real-time.
"""


def launch_dashboard(
    storage: str,
    port: int = 8080,
    host: str = "127.0.0.1",
    open_browser: bool = True,
) -> None:
    """
    Launch the Optuna Dashboard to monitor optimization studies.

    This function is designed to be run from a separate terminal while
    optimization is running in another process (e.g., a notebook).

    Parameters
    ----------
    storage : str
        Database URL for the Optuna study
        (e.g., "sqlite:///optuna_study.db").
    port : int, default=8080
        Port number for the dashboard server.
    host : str, default="127.0.0.1"
        Host address for the dashboard server.
    open_browser : bool, default=True
        Whether to automatically open the dashboard in a web browser.

    Examples
    --------
    From Python:

    >>> from rctbp_bf_training.core.dashboard import launch_dashboard
    >>> launch_dashboard("sqlite:///optuna_ancova_cont_2arms.db")

    From command line:

    >>> python -m rctbp_bf_training.core.dashboard --dashboard sqlite:///study.db

    Notes
    -----
    Requires optuna-dashboard package: pip install optuna-dashboard
    The dashboard provides real-time visualization of:
    - Trial history and progress
    - Pareto front for multi-objective optimization
    - Hyperparameter importance
    - Parallel coordinate plots
    """
    try:
        import optuna_dashboard
    except ImportError:
        print("\u274c optuna-dashboard not installed.")
        print("   Install with: pip install optuna-dashboard")
        return

    import webbrowser

    url = f"http://{host}:{port}"
    print("\U0001f680 Starting Optuna Dashboard...")
    print(f"   Storage: {storage}")
    print(f"   URL: {url}")
    print("\n   Press Ctrl+C to stop the server.")

    if open_browser:
        import threading
        import time

        def open_browser_delayed() -> None:
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(
            target=open_browser_delayed, daemon=True
        ).start()

    # Run the dashboard (blocking)
    optuna_dashboard.run_server(storage, host=host, port=port)


def _cli_main() -> None:
    """Command-line interface for dashboard utilities."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Optuna dashboard utilities for rctbp_bf_training"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Launch dashboard:
    python -m rctbp_bf_training.core.dashboard --dashboard sqlite:///study.db
    python -m rctbp_bf_training.core.dashboard \
      --dashboard sqlite:///study.db --port 8888
        """,
    )
    parser.add_argument(
        "--dashboard",
        metavar="STORAGE",
        help="Launch Optuna dashboard with the specified storage URL",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the dashboard server (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the dashboard server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard(
            storage=args.dashboard,
            port=args.port,
            host=args.host,
            open_browser=not args.no_browser,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
