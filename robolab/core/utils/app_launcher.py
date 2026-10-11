"""Keep RoboLab's launch flags stable across Isaac Lab releases."""


def add_app_launcher_args(parser):
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    destinations = {action.dest for action in parser._actions}
    # Lab release/3.0 removed these CLI flags in favor of --viz, but still
    # accepts the corresponding AppLauncher kwargs. Existing deploy scripts
    # and the shared evaluator use them; add aliases only when absent upstream.
    if "headless" not in destinations:
        parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without a display window.")
    if "enable_cameras" not in destinations:
        parser.add_argument("--enable_cameras", "--enable-cameras", action="store_true",
                            help="Enable camera rendering (legacy launch option).")
