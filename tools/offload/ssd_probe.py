"""Random reads of one expert's size (1.19 MiB), O_DIRECT so the page cache is out of it, at queue depths 1 and 8."""
import glob, mmap, os, random, sys, threading, time
files = sorted(glob.glob(sys.argv[1] + "/*.safetensors"))
SIZE = (int(1.19 * 2**20) // 4096 + 1) * 4096
sizes = [os.path.getsize(f) for f in files]
def worker(n, out, seed):
    r = random.Random(seed); buf = mmap.mmap(-1, SIZE)
    fds = [os.open(f, os.O_RDONLY | os.O_DIRECT) for f in files]
    t = time.time()
    for _ in range(n):
        k = r.randrange(len(files)); off = r.randrange(0, (sizes[k] - SIZE) // 4096) * 4096
        os.preadv(fds[k], [buf], off)
    out.append(time.time() - t)
for qd in (1, 8):
    n = 400 if qd == 1 else 200
    out = []; th = [threading.Thread(target=worker, args=(n, out, qd * 100 + i)) for i in range(qd)]
    t = time.time(); [x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t
    reads = n * qd
    print(f"QD{qd}: {reads} reads of {SIZE / 2**20:.2f} MiB in {wall:.2f} s: {reads * SIZE / wall / 1e9:.2f} GB/s, "
          f"{wall / reads * 1e3 * qd:.2f} ms per read at that depth")
