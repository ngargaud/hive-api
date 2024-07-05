#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from hive_api import HiveApi

def main():
    """
        Interact with the Hive.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        help="Display additionnal informations")
    parser.add_argument("--hive", type=str,
                        help="Hive address with protocol (without port). Do not set if using the hive notebook.")
    args = parser.parse_args()

    os.environ["HTTPX_NO_SSL"] = "True"
    hive = HiveApi(url=args.hive)

    print(hive.get_api_settings("asr"))
    print(hive.get_api_settings("tts"))
    print(hive.get_api_settings("voicereco"))
    print(hive.get_api_settings("facereco"))
    print(json.dumps(hive.get_client("ollama").list(), indent=2))

main()
