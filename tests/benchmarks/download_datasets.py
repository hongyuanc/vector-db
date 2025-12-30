#!/usr/bin/env python3
"""
Download benchmark datasets.

Usage:
    python download_datasets.py --all           # Download all datasets
    python download_datasets.py --sift          # Download SIFT1M only
    python download_datasets.py --list          # List available datasets
"""

import argparse
from real_datasets import download_dataset, list_available_datasets, DATASETS


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--sift", action="store_true", help="Download SIFT1M")
    parser.add_argument("--gist", action="store_true", help="Download GIST1M")
    parser.add_argument("--glove", action="store_true", help="Download GloVe")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")

    args = parser.parse_args()

    if args.list:
        list_available_datasets()
        return

    if not any([args.all, args.sift, args.gist, args.glove]):
        print("Please specify which dataset(s) to download.")
        print("Use --help for options.")
        list_available_datasets()
        return

    # Download requested datasets
    if args.all or args.sift:
        print("\n" + "="*60)
        print("Downloading SIFT1M")
        print("="*60)
        print("Size: ~175 MB compressed, ~500 MB uncompressed")
        print("This may take a few minutes...\n")
        try:
            download_dataset("sift1m", force=args.force)
        except Exception as e:
            print(f"Failed to download SIFT1M: {e}")

    if args.all or args.gist:
        print("\n" + "="*60)
        print("Downloading GIST1M")
        print("="*60)
        print("Size: ~3 GB compressed")
        print("This may take 10-30 minutes depending on connection...\n")
        try:
            download_dataset("gist1m", force=args.force)
        except Exception as e:
            print(f"Failed to download GIST1M: {e}")

    if args.all or args.glove:
        print("\n" + "="*60)
        print("Downloading GloVe")
        print("="*60)
        print("Size: ~822 MB")
        print("This may take a few minutes...\n")
        try:
            download_dataset("glove-50d", force=args.force)
        except Exception as e:
            print(f"Failed to download GloVe: {e}")

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print("\nTo run benchmarks:")
    print("  pytest tests/benchmarks/test_real_world.py -v -s -m real_data")


if __name__ == "__main__":
    main()
