"""Build and test logs from real open-source projects, for measuring (and possibly calibrating
on) the log and error output agents read all day. Each project is cloned, then built and tested
three times in its ecosystem's official container image: clean, with a behavioral fault (a
flipped comparison: test failures), and with a name fault (a misspelled identifier: compiler
errors or tracebacks). Output is a terminal transcript per project ("$ cmd" then its output).

Usage: python logcorpus.py OUT_DIR [project ...]. Needs rootless podman and network. Projects
are permissively licensed; per ecosystem the last listed is the eval split, the rest calibration.
"""
import json, os, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ECO = {
    "python": dict(image = "docker.io/library/python:3.12", exts = (".py",), cache = {"pip": "/root/.cache/pip"},
                   setup = "for f in requirements*test*.txt requirements/test*.txt test-requirements.txt requirements-dev.txt; do [ -f $f ] && pip install -q --progress-bar off -r $f; done",
                   build = "pip install --progress-bar off -e . pytest",
                   test = "python -m pytest -p no:cacheprovider"),
    "c": dict(image = "docker.io/library/gcc:14", exts = (".c", ".cc", ".cpp", ".h"), cache = {},
              setup = "apt-get -qq update && apt-get -qq install -y cmake",
              build = "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j4",
              test = "ctest --test-dir build --output-on-failure"),
    "rust": dict(image = "docker.io/library/rust:1", exts = (".rs",), cache = {"cargo": "/usr/local/cargo/registry"},
                 build = "cargo build --all-targets", test = "cargo test"),
    "go": dict(image = "docker.io/library/golang:1", exts = (".go",), cache = {"gomod": "/go/pkg/mod"},
               build = "go build ./... && go vet ./...", test = "go test ./..."),
    "node": dict(image = "docker.io/library/node:22", exts = (".js", ".mjs", ".ts"), cache = {"npm": "/root/.npm"},
                 build = "npm install --no-audit --no-fund", test = "npm test"),
    "java": dict(image = "docker.io/library/maven:3-eclipse-temurin-21", exts = (".java",), cache = {"m2": "/root/.m2"},
                 build = "mvn -B -ntp -DskipTests compile", test = "mvn -B -ntp test"),
}
PROJECTS = {   # (name, repo); last per ecosystem = eval
    "python": [("click", "https://github.com/pallets/click"), ("attrs", "https://github.com/python-attrs/attrs"),
               ("more-itertools", "https://github.com/more-itertools/more-itertools"), ("toolz", "https://github.com/pytoolz/toolz")],
    "c": [("cJSON", "https://github.com/DaveGamble/cJSON"), ("jansson", "https://github.com/akheron/jansson"),
          ("fmt", "https://github.com/fmtlib/fmt"), ("zlib", "https://github.com/madler/zlib")],
    "rust": [("log", "https://github.com/rust-lang/log"), ("bitflags", "https://github.com/bitflags/bitflags"),
             ("semver", "https://github.com/dtolnay/semver"), ("itoa", "https://github.com/dtolnay/itoa")],
    "go": [("pflag", "https://github.com/spf13/pflag"), ("mux", "https://github.com/gorilla/mux"),
           ("logrus", "https://github.com/sirupsen/logrus"), ("uuid", "https://github.com/google/uuid")],
    "node": [("express", "https://github.com/expressjs/express"), ("debug", "https://github.com/debug-js/debug"),
             ("commander", "https://github.com/tj/commander.js"), ("yargs", "https://github.com/yargs/yargs")],
    "java": [("gson", "https://github.com/google/gson"), ("commons-cli", "https://github.com/apache/commons-cli"),
             ("commons-csv", "https://github.com/apache/commons-csv"), ("jsoup", "https://github.com/jhy/jsoup")],
}

def sh(cmd, **kw):
    return subprocess.run(cmd, shell = isinstance(cmd, str), capture_output = True, text = True, errors = "replace", **kw)

def sources(src, exts):
    out = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "target", "build", "vendor", "docs", "test", "tests", "testdata", "__tests__",
                                                                            "fuzzing", "fuzz", "examples", "example", "bench", "benches", "benchmark")]
        out += [os.path.join(root, f) for f in files if f.endswith(exts) and "test" not in f.lower()]
    return sorted(out)

def mutate(src, exts, rng, kind):
    """Apply one fault in a non-test source file; returns a description, or None."""
    files = sources(src, exts)
    rng.shuffle(files)
    for f in files:
        lines = open(f, errors = "replace").read().split("\n")
        if kind == "behavior":
            cand = [i for i, l in enumerate(lines) if re.search(r" (==|<|>|<=|>=|!=) ", l) and not l.strip().startswith(("//", "#", "*"))]
            if not cand: continue
            i = rng.choice(cand)
            flip = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}
            lines[i] = re.sub(r" (==|<|>|<=|>=|!=) ", lambda m: f" {flip[m.group(1)]} ", lines[i], count = 1)
        else:
            text = "\n".join(lines)
            names = [w for w in set(re.findall(r"\b[a-z_][A-Za-z0-9_]{4,}\b", text)) if text.count(w) >= 3]
            cand = [(i, w) for i, l in enumerate(lines) for w in names if re.search(rf"\b{w}\b", l) and not l.strip().startswith(("//", "#", "*", "import", "from", "use "))]
            if not cand: continue
            i, w = rng.choice(cand)
            lines[i] = re.sub(rf"\b{w}\b", w[:-1] + w[-1] * 2, lines[i], count = 1)
        open(f, "w").write("\n".join(lines))
        return f"{os.path.relpath(f, src)}:{i + 1}"
    return None

def run_in(eco, src, script, timeout = 1800):
    e = ECO[eco]
    vols = sum((["-v", f"exl3logs-{eco}-{k}:{p}"] for k, p in e["cache"].items()), [])
    r = sh(["podman", "run", "--rm", "-v", f"{src}:/work/src:Z", "-w", "/work/src", *vols, e["image"], "bash", "-c", script], timeout = timeout)
    return r.stdout

def transcript(eco, cmds):
    # One container run per phase; each command echoed as a prompt line, stderr folded into stdout;
    # the ecosystem's setup runs first, silently
    setup = ECO[eco].get("setup")
    return (f"{{ {setup} ; }} >/dev/null 2>&1\n" if setup else "") + "\n".join(f"echo '$ {c.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'; {{ {c} ; }} 2>&1" for c in cmds)

def project(out, eco, name, repo, split):
    rng = random.Random(name)
    src = os.path.join(out, "src", name)
    if not os.path.isdir(src):
        sh(["git", "clone", "-q", "--depth", "1", repo, src])
    commit = sh(["git", "-C", src, "rev-parse", "HEAD"]).stdout.strip()
    e, log, faults = ECO[eco], [], []
    t0 = time.time()
    for phase in ("clean", "behavior", "name"):
        sh(["git", "-C", src, "checkout", "-q", "--", "."])
        if phase != "clean":
            where = mutate(src, e["exts"], rng, phase)
            if not where: continue
            faults.append({"phase": phase, "at": where})
            diff = sh(["git", "-C", src, "diff"]).stdout
            log.append(f"$ git diff\n{diff}")
        try:
            log.append(run_in(eco, src, transcript(eco, [e["build"], e["test"]])))
        except subprocess.TimeoutExpired:
            log.append(f"[timed out]")
    sh(["git", "-C", src, "checkout", "-q", "--", "."])
    text = "\n".join(log)
    open(os.path.join(out, f"{eco}__{name}.log"), "w").write(text)
    print(f" -- {eco}/{name} ({split}): {len(text) // 1000} k chars, {time.time() - t0:.0f} s, faults {faults}", flush = True)
    return {"eco": eco, "name": name, "repo": repo, "commit": commit, "split": split, "faults": faults, "chars": len(text)}

def main():
    out = os.path.abspath(sys.argv[1]); os.makedirs(os.path.join(out, "src"), exist_ok = True)
    only = set(sys.argv[2:])
    jobs = [(eco, n, r, "eval" if k == len(ps) - 1 else "cal") for eco, ps in PROJECTS.items() for k, (n, r) in enumerate(ps)
            if not only or n in only]
    for eco in {j[0] for j in jobs}:
        sh(["podman", "pull", "-q", ECO[eco]["image"]])
    with ThreadPoolExecutor(4) as ex:
        man = list(ex.map(lambda j: project(out, *j), jobs))
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent = 1)

if __name__ == "__main__":
    main()
