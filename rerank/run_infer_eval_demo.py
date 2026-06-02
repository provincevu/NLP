import argparse
import os
import subprocess


def main():
    p = argparse.ArgumentParser(description="Quick demo: infer then evaluate")
    p.add_argument("--input_path", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_path", default=None)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    output_path = args.output_path
    if output_path is None:
        base, ext = os.path.splitext(args.input_path)
        if ext.lower() == ".jsonl":
            output_path = base + ".scored.jsonl"
        else:
            output_path = base + ".scored.json"

    infer_cmd = [
        "python",
        "rerank/infer_cross_encoder.py",
        "--input_path",
        args.input_path,
        "--output_path",
        output_path,
        "--ckpt_dir",
        args.ckpt_dir,
        "--batch_size",
        str(args.batch_size),
        "--max_len",
        str(args.max_len),
        "--device",
        args.device,
        "--save_logit",
        "--save_prob",
    ]
    eval_cmd = [
        "python",
        "rerank/evaluate_cross_encoder.py",
        "--input_path",
        output_path,
        "--score_key",
        "ce_logit",
        "--score_type",
        "logit",
    ]

    print("Running inference...")
    subprocess.check_call(infer_cmd)
    print("Running evaluation...")
    subprocess.check_call(eval_cmd)


if __name__ == "__main__":
    main()
