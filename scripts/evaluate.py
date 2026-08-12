import argparse
import json
from time import sleep
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def call_api(base_url: str, question: str) -> dict:
    payload = json.dumps({"question": question}).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JSON self-test against the HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default="sample_test_cases.json")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    passed = 0
    for case in cases:
        try:
            result = None
            attempts = 3 if case.get("type") == "in_corpus" else 1
            for attempt in range(attempts):
                result = call_api(args.base_url, case["question"])
                actual_files = {citation["source_file"] for citation in result.get("citations", [])}
                expected_files = set(
                    case.get("expected_citation_files") or case.get("expected_source_files") or []
                )
                expected_grounded = case.get("expected_grounded")
                if expected_grounded is None:
                    expected_grounded = case.get("type") == "in_corpus"
                if case.get("type") != "in_corpus":
                    break
                if result.get("grounded") == expected_grounded and expected_files.issubset(actual_files):
                    break
                if attempt < attempts - 1:
                    sleep(2)
            actual_files = {citation["source_file"] for citation in result.get("citations", [])}
            ok = (
                result.get("grounded") == expected_grounded
                and expected_files.issubset(actual_files)
                and (expected_files or not actual_files)
            )
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"{status} {case['id']}: {result.get('answer', '')}")
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"FAIL {case['id']}: HTTP request failed: {exc}")

    print(f"\n{passed}/{len(cases)} cases passed")
    raise SystemExit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()

