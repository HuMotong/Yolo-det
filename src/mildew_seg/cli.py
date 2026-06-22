from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import __version__
from .audit import run_audit
from .batch import run_batch
from .calibrate import run_calibration
from .config import load_config, path_from_config, resolve_path
from .decision import DecisionThresholds
from .demo import launch_demo
from .inference import MildewPredictor, save_prediction
from .prepare import prepare_dataset
from .thresholds import load_thresholds
from .train import run_training
from .utils import LOGGER, safe_name, setup_logging, timestamp
from .validate import run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mildew-seg",
        description="YOLO11-seg seed mildew spot project",
    )
    parser.add_argument("--version", action="version", version=__version__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML configuration path",
    )
    common.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", parents=[common], help="Audit LabelMe data")
    check.set_defaults(handler=_handle_check)

    prepare = subparsers.add_parser(
        "prepare", parents=[common], help="Build tiled YOLO-seg dataset"
    )
    prepare.add_argument("--no-clean", action="store_true")
    prepare.set_defaults(handler=_handle_prepare)

    train = subparsers.add_parser(
        "train", parents=[common], help="Fine-tune YOLO11-seg"
    )
    train.add_argument("--model", help="Initial checkpoint")
    train.add_argument("--data", help="Dataset YAML")
    train.add_argument("--name", default=None, help="Ultralytics run name")
    train.add_argument("--resume", action="store_true")
    train.set_defaults(handler=_handle_train)

    validate = subparsers.add_parser(
        "validate", parents=[common], help="Run slice and original-image validation"
    )
    validate.add_argument("--model", help="Trained checkpoint")
    validate.add_argument("--data", help="Dataset YAML")
    validate.add_argument("--manifest", help="Prepared manifest CSV")
    validate.add_argument("--thresholds", help="Calibrated thresholds YAML")
    validate.add_argument("--split", choices=["val", "test"], default="test")
    validate.set_defaults(handler=_handle_validate)

    predict = subparsers.add_parser(
        "predict", parents=[common], help="Predict one image"
    )
    predict.add_argument("--source", required=True)
    predict.add_argument("--model", help="Trained checkpoint")
    predict.add_argument("--thresholds", help="Calibrated thresholds YAML")
    predict.set_defaults(handler=_handle_predict)

    batch = subparsers.add_parser(
        "batch", parents=[common], help="Predict a file or image directory"
    )
    batch.add_argument("--source", required=True)
    batch.add_argument("--model", help="Trained checkpoint")
    batch.add_argument("--thresholds", help="Calibrated thresholds YAML")
    batch.add_argument("--no-images", action="store_true")
    batch.set_defaults(handler=_handle_batch)

    calibrate = subparsers.add_parser(
        "calibrate", parents=[common], help="Calibrate image decision thresholds"
    )
    calibrate.add_argument("--model", help="Trained checkpoint")
    calibrate.add_argument("--manifest", help="Prepared manifest CSV")
    calibrate.set_defaults(handler=_handle_calibrate)

    demo = subparsers.add_parser("demo", parents=[common], help="Launch Gradio demo")
    demo.add_argument("--model", help="Trained checkpoint")
    demo.add_argument("--thresholds", help="Calibrated thresholds YAML")
    demo.add_argument("--server-name", default="127.0.0.1")
    demo.add_argument("--server-port", type=int, default=7860)
    demo.add_argument("--share", action="store_true")
    demo.set_defaults(handler=_handle_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    runs_root = path_from_config(config, "runs")
    log_dir = runs_root / args.command
    setup_logging(log_dir, args.command, args.verbose)
    LOGGER.debug("Arguments: %s", vars(args))
    args.handler(args, config)
    return 0


def _handle_check(args: argparse.Namespace, config: dict[str, Any]) -> None:
    output = path_from_config(config, "runs") / "data"
    summary = run_audit(config, output, allow_repair=True)
    LOGGER.info("Audit report: %s", output / "audit_report.json")
    if summary["severity_counts"].get("error", 0):
        raise SystemExit(2)


def _handle_prepare(args: argparse.Namespace, config: dict[str, Any]) -> None:
    prepare_dataset(
        config,
        path_from_config(config, "runs") / "data",
        clean=not args.no_clean,
    )


def _handle_train(args: argparse.Namespace, config: dict[str, Any]) -> None:
    model_path = _model_path(args.model, config, prefer_trained=False)
    data_yaml = _argument_path(
        args.data, config, path_from_config(config, "runs") / "data" / "data.yaml"
    )
    name = args.name or f"yolo11n_seg_{timestamp()}"
    run_training(
        config,
        model_path,
        data_yaml,
        path_from_config(config, "runs") / "train",
        name,
        resume=args.resume,
    )


def _handle_validate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    model_path = _model_path(args.model, config, prefer_trained=True)
    data_yaml = _argument_path(
        args.data, config, path_from_config(config, "runs") / "data" / "data.yaml"
    )
    manifest = _argument_path(
        args.manifest,
        config,
        path_from_config(config, "runs") / "data" / "manifest.csv",
    )
    thresholds = _thresholds(args.thresholds, config)
    predictor = MildewPredictor(model_path, config)
    run_validation(
        predictor,
        data_yaml,
        manifest,
        path_from_config(config, "runs") / "val" / args.split,
        config,
        thresholds,
        split=args.split,
    )


def _handle_predict(args: argparse.Namespace, config: dict[str, Any]) -> None:
    source = resolve_path(config, args.source)
    predictor = MildewPredictor(
        _model_path(args.model, config, prefer_trained=True), config
    )
    result, visualization = predictor.predict_path(
        source, _thresholds(args.thresholds, config)
    )
    image_path, json_path = save_prediction(
        result,
        visualization,
        path_from_config(config, "runs") / "predict",
        safe_name(source.stem),
    )
    LOGGER.info(
        "Decision=%s image=%s result=%s", result.decision.label, image_path, json_path
    )


def _handle_batch(args: argparse.Namespace, config: dict[str, Any]) -> None:
    predictor = MildewPredictor(
        _model_path(args.model, config, prefer_trained=True), config
    )
    run_batch(
        predictor,
        resolve_path(config, args.source),
        path_from_config(config, "runs") / "batch",
        config,
        _thresholds(args.thresholds, config),
        save_images=not args.no_images,
    )


def _handle_calibrate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    predictor = MildewPredictor(
        _model_path(args.model, config, prefer_trained=True), config
    )
    manifest = _argument_path(
        args.manifest,
        config,
        path_from_config(config, "runs") / "data" / "manifest.csv",
    )
    run_calibration(
        predictor,
        manifest,
        path_from_config(config, "runs") / "calibration",
        config,
    )


def _handle_demo(args: argparse.Namespace, config: dict[str, Any]) -> None:
    predictor = MildewPredictor(
        _model_path(args.model, config, prefer_trained=True), config
    )
    calibrated = _thresholds(args.thresholds, config)
    config["decision"] = {
        "instance_confidence": calibrated.instance_confidence,
        "mean_confidence": calibrated.mean_confidence,
        "count_threshold": calibrated.count_threshold,
        "area_ratio_threshold": calibrated.area_ratio_threshold,
    }
    launch_demo(
        predictor,
        path_from_config(config, "runs") / "demo",
        config,
        args.server_name,
        args.server_port,
        args.share,
    )


def _model_path(
    argument: str | None,
    config: dict[str, Any],
    prefer_trained: bool,
) -> Path:
    if argument:
        return resolve_path(config, argument)
    if prefer_trained:
        candidates = sorted(
            (path_from_config(config, "runs") / "train").glob("**/weights/best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
    return path_from_config(config, "checkpoint")


def _argument_path(
    argument: str | None,
    config: dict[str, Any],
    default: Path,
) -> Path:
    return resolve_path(config, argument) if argument else default.resolve()


def _thresholds(
    argument: str | None,
    config: dict[str, Any],
) -> DecisionThresholds:
    defaults = DecisionThresholds.from_config(config)
    if argument:
        path = resolve_path(config, argument)
    else:
        calibrated = (
            path_from_config(config, "runs") / "calibration" / "thresholds.yaml"
        )
        path = calibrated if calibrated.is_file() else None
    return load_thresholds(path, defaults)


if __name__ == "__main__":
    raise SystemExit(main())
