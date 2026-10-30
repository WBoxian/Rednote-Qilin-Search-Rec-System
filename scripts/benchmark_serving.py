"""测量真实 HTTP 延迟；将客户端耗时、服务端计时与缓存状态分别记录。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np

def request(base, case, timeout):
    started=time.perf_counter()
    row={"case":case}
    try:
        with urlopen(base.rstrip("/")+"/api/feed?"+urlencode(case),timeout=timeout) as response:
            body=json.load(response)
        if not body.get("ok"):
            raise ValueError(str(body.get("detail",body)))
        feed=body.get("feed",{})
        row.update(ok=True,server_ms=body.get("latency_ms"),cache_hit=feed.get("cache_hit"),
                   pipeline_ms=feed.get("pipeline_ms"),stages=feed.get("stages"),
                   items=len(feed.get("items",[])))
    except Exception as exc:
        row.update(ok=False,error=str(exc))
    row["client_ms"]=(time.perf_counter()-started)*1000
    return row

def summarize(rows):
    successful=[row["client_ms"] for row in rows if row["ok"]]
    return {"requests":len(rows),"errors":sum(not row["ok"] for row in rows),
            "latency_ms":dict(zip(("p50","p95","p99","max"),
                [float(x) for x in np.percentile(successful,[50,95,99,100])])) if successful else None,
            "cache_hits":sum(row.get("cache_hit") is True for row in rows)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",default="http://127.0.0.1:18082")
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--scene",choices=("search","rec"),required=True)
    parser.add_argument("--samples",type=int,default=20)
    parser.add_argument("--concurrency",type=int,default=1)
    parser.add_argument("--timeout",type=float,default=180)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.samples<1 or args.concurrency<1: parser.error("samples/concurrency must be positive")
    cases=[case for case in json.loads(args.cases.read_text()) if case.get("scene")==args.scene]
    if not cases: parser.error("cases cannot be empty")
    first=[]
    for case in cases:
        row=request(args.base_url,case,args.timeout)
        first.append(row)
        print(json.dumps({"phase":"first_observation",**row},ensure_ascii=False),flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        warm=list(pool.map(lambda index:request(args.base_url,cases[index%len(cases)],args.timeout),range(args.samples)))
    output={"scene":args.scene,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "platform":platform.platform(),"base_url":args.base_url,"concurrency":args.concurrency,
            "first_observations":first,"warm_summary":summarize(warm),"warm_requests":warm,
            "limitations":["First observation is cold only when server/runtime was freshly started",
                           "Fixed-case repeated traffic measures cache-heavy workload, not general production capacity"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(output["warm_summary"],ensure_ascii=False),flush=True)
    if any(not row["ok"] for row in first+warm): raise SystemExit(1)

if __name__=="__main__": main()
