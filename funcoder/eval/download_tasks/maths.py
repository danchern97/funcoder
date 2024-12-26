import json
import pathlib
import tarfile

import requests
import tqdm


def download_maths_dataset(to_dir: pathlib.Path, cfg_proxy: str | None = None) -> None:
    print("[MATH dataset]")
    to_dir.mkdir(parents=True, exist_ok=True)

    # download
    the_tar_url = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"
    the_tar_path = to_dir / "MATH.tar"
    if not the_tar_path.exists():
        print(f"  - downloading file: {the_tar_url}")
        req = requests.get(the_tar_url, proxies={"http": cfg_proxy, "https": cfg_proxy} if cfg_proxy else None)
        with open(the_tar_path, "wb") as f:
            f.write(req.content)
    else:
        print(f"  - file already exists: {to_dir.name}/MATH.tar")

    # extract
    print(f"  - extracting tar: {to_dir.name}/MATH.tar")
    with tarfile.open(the_tar_path) as the_tar:
        the_tar.extractall(to_dir)

    # two splits
    for split in ["test", "train"]:
        input_dir = to_dir / f"MATH/{split}"
        output_file = to_dir / f"MATH_{split}.json"
        print(f"  - collating {to_dir.name}/MATH/{split} -> {to_dir.name}/MATH_{split}.json")

        # know the tasks
        files: list[tuple[str, int, str]] = []  # (task, num, content)
        for filename in tqdm.tqdm(list(input_dir.glob("**/*.json"))):
            task, num = filename.parent.name, int(filename.stem)
            with open(filename, "r", encoding="utf-8") as f:
                files.append((task, num, f.read()))
        files.sort(key=lambda x: (x[0], x[1]))
        print(f"  - discovered {len(files)} tasks")

        # compress into a json file, but that every dict is on one line
        output: list[str] = ["{"]
        for task, num, content in tqdm.tqdm(files):
            compressed = json.dumps(json.loads(content), indent=None, ensure_ascii=False)
            if len(output) == 1:
                output.append(f'\n  "{task}/{num}": {compressed}')
            else:
                output.append(f',\n  "{task}/{num}": {compressed}')
        output.append("\n}\n")

        # write output
        print(f"  - writing to {output_file.name}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(output))

    print("  - done")
    return
