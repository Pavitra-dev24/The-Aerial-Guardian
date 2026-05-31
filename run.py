import argparse
import os
import sys
from src.pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Aerial Guardian - Drone person detection and MOT pipeline"
    )
    p.add_argument("input", help="Input video file or image sequence directory")
    p.add_argument("output", help="Output video path (.mp4)")
    p.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    p.add_argument("--device", default=None, help="Inference device: cpu, cuda, mps")
    return p.parse_args()


def main():
    args = parse_args()

    print("Aerial Guardian")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Config : {args.config}")
    print()

    pipeline = Pipeline(config_path=args.config)

    if args.device:
        pipeline.set_device(args.device)
        print(f"  Device override: {args.device}")

    if os.path.isdir(args.input):
        print(f"Mode: image sequence")
        pipeline.process_sequence(args.input, args.output)
    elif os.path.isfile(args.input):
        print(f"Mode: video file")
        pipeline.process_video(args.input, args.output)
    else:
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
