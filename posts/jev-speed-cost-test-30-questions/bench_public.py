# -*- coding: utf-8 -*-
"""11/02 월요일 롱폼 실험 — '200배 빠르고 400배 싸다'는 Jev, 같은 고객 문의 분류 30문항으로 다시 재 보기.
모델 4개(Jev / GPT-6 Luna / Gemini 3.5 Flash-Lite / Claude Haiku 5.5)에 같은 지시문·같은 라벨 5종·같은 문항을
문항 1개당 호출 1번씩, 3회차 반복(회차마다 모델 순서와 문항 순서를 바꿈, 고정 시드). 워밍업 1회는 결과에서 뺌.
시간 = 요청 직전~응답 전체 수신(time.perf_counter), 같은 PC·같은 회선, 순차 호출, 호스트별 연결 재사용(keep-alive).
키는 환경 변수에서 읽고 화면·파일에 찍지 않는다(새AI실험노트 공개본, 2026-11-02).
사용: python bench.py --probe        (모델별 1번씩 호출 확인, 기록 안 함)
      python bench.py --run [--record] (본 실험, results/ 에 저장)"""
import sys, os, json, time, random, hashlib, statistics, datetime, http.client, ssl, subprocess, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sys.path.insert(0, str(HERE))
from items import LABELS, INSTR, ITEMS

try:
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    CTX = ssl.create_default_context()
ENV = os.environ  # 공개본: 키는 환경 변수(TYPESAFE_API_KEY·OPENAI_API_KEY·GEMINI_API_KEY·ANTHROPIC_API_KEY)에서 읽음
OUT = HERE / "results"; REC = HERE / "recordings"; TITLE = "AI-SPEED-BENCH"
C = {"g": "\033[92m", "r": "\033[91m", "y": "\033[93m", "c": "\033[96m", "m": "\033[95m", "d": "\033[90m", "b": "\033[1m", "w": "\033[97m", "0": "\033[0m"}
NAMES = list(LABELS)
SYS = INSTR + "\n" + "\n".join(f"- {k}: {v}" for k, v in LABELS.items()) + '\n답은 {"label": "분류 이름"} 형식의 JSON 하나만 출력하세요.'
PRICE = {"jev": (0.042, 0.0), "gpt": (0.10, 0.50), "gemini": (0.30, 2.50), "claude": (0.10, 0.50), "sol": (2.0, 10.0), "fable": (10.0, 50.0)}  # 달러 / 100만 토큰 (입력, 출력) — 11/02 공식 문서
MODEL = {"jev": "jev-latest", "gpt": "gpt-6-luna", "gemini": "gemini-3.5-flash-lite", "claude": "claude-haiku-5-5"}
BIG = {"sol": "gpt-6.1-sol", "fable": "claude-fable-5-1"}  # 큰 모델 추가 비교(출처가 GPT·Fable과 비교) — 30문항 1회차만, 기본 설정
KIND = {"jev": "jev", "gpt": "oa", "sol": "oa", "gemini": "gem", "claude": "an", "fable": "an"}
SHOW = {"jev": "Jev", "gpt": "GPT-6 Luna", "gemini": "Gemini 3.5 FL", "claude": "Claude Haiku 5.5", "sol": "GPT-6.1 Sol", "fable": "Claude Fable 5.1"}
COL = {"jev": "m", "gpt": "g", "gemini": "c", "claude": "y", "sol": "g", "fable": "y"}
def mid(m): return MODEL.get(m) or BIG[m]
REC_T0 = [None]
def roff(): return round(time.time() - REC_T0[0], 2) if REC_T0[0] else None
def kst(): return (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")

CONN = {}
def post(host, path, body, headers, timeout=60):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json", "Connection": "keep-alive"}; h.update(headers)
    for attempt in (1, 2):
        c = CONN.get(host)
        if c is None:
            c = CONN[host] = http.client.HTTPSConnection(host, timeout=timeout, context=CTX)
        try:
            c.request("POST", path, body=data, headers=h); r = c.getresponse(); raw = r.read()
            return r.status, raw
        except (http.client.HTTPException, OSError):
            try: c.close()
            except Exception: pass
            CONN.pop(host, None)
            if attempt == 2: raise
    raise RuntimeError("unreachable")

def strict_parse(text):
    try:
        v = json.loads(text)
        if isinstance(v, dict) and v.get("label") in LABELS: return v["label"]
    except Exception: pass
    return None
def loose_parse(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[4:] if t.lower().startswith("json") else t
    return strict_parse(t.strip())

def call(m, q):
    """반환: dict(raw, label, loose, tin, tout, status, note)"""
    if m == "jev":
        st, raw = post("api.typesafe.ai", "/v1/systemone",
                       {"state": q, "model": MODEL["jev"], "questions": {"label": {"type": "choice", "instructions": INSTR, "criteria": LABELS}}},
                       {"Authorization": "Bearer " + ENV["TYPESAFE_API_KEY"]})
        j = json.loads(raw); a = (j.get("answers") or {}).get("label") or {}
        lab = a.get("choice") if a.get("choice") in LABELS else None
        u = j.get("usage") or {}
        return {"raw": json.dumps(a, ensure_ascii=False), "label": lab, "loose": lab, "tin": u.get("input_tokens"), "tout": u.get("output_tokens"),
                "status": st, "note": j.get("model"), "prob": (a.get("probabilities") or {}).get(lab) if lab else None}
    if KIND[m] == "oa":
        body = {"model": mid(m), "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "cls", "strict": True, "schema": {
                    "type": "object", "properties": {"label": {"type": "string", "enum": NAMES}}, "required": ["label"], "additionalProperties": False}}},
                "max_completion_tokens": 4000}
        if m == "gpt": body["reasoning_effort"] = GPT_EFFORT[0]
        st, raw = post("api.openai.com", "/v1/chat/completions", body, {"Authorization": "Bearer " + ENV["OPENAI_API_KEY"]})
        j = json.loads(raw)
        if st != 200: return {"raw": raw.decode("utf-8", "replace")[:300], "label": None, "loose": None, "tin": None, "tout": None, "status": st, "note": "error"}
        txt = j["choices"][0]["message"].get("content") or ""; u = j.get("usage") or {}
        rt = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        return {"raw": txt, "label": strict_parse(txt), "loose": loose_parse(txt), "tin": u.get("prompt_tokens"), "tout": u.get("completion_tokens"),
                "status": st, "note": f"{j.get('model')} effort={GPT_EFFORT[0] if m == 'gpt' else '기본값'} reasoning_tokens={rt}"}
    if m == "gemini":
        body = {"systemInstruction": {"parts": [{"text": SYS}]}, "contents": [{"role": "user", "parts": [{"text": q}]}],
                "generationConfig": {"responseMimeType": "application/json", "responseSchema": {"type": "OBJECT", "properties": {"label": {"type": "STRING", "enum": NAMES}}, "required": ["label"]},
                                     "temperature": 0, "maxOutputTokens": 300, "thinkingConfig": GEM_THINK[0]}}
        st, raw = post("generativelanguage.googleapis.com", f"/v1beta/models/{MODEL['gemini']}:generateContent", body, {"x-goog-api-key": ENV["GEMINI_API_KEY"]})
        j = json.loads(raw)
        if st != 200: return {"raw": raw.decode("utf-8", "replace")[:300], "label": None, "loose": None, "tin": None, "tout": None, "status": st, "note": "error"}
        parts = (((j.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
        txt = "".join(p.get("text", "") for p in parts if not p.get("thought")); u = j.get("usageMetadata") or {}
        tout = (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
        return {"raw": txt, "label": strict_parse(txt), "loose": loose_parse(txt), "tin": u.get("promptTokenCount"), "tout": tout,
                "status": st, "note": f"{j.get('modelVersion')} thinking={json.dumps(GEM_THINK[0])} thoughts={u.get('thoughtsTokenCount')}"}
    if KIND[m] == "an":
        body = {"model": mid(m), "max_tokens": 2000, "system": SYS, "messages": [{"role": "user", "content": q}],
                "tools": [{"name": "classify", "description": "고객 문의 분류 결과", "input_schema": {"type": "object", "properties": {"label": {"type": "string", "enum": NAMES}}, "required": ["label"]}}],
                "tool_choice": {"type": "tool", "name": "classify"}}
        if m == "fable":  # Fable 5.1은 강제 도구 호출(tool_choice tool/any) 미지원 → 구조화 출력(JSON 스키마)으로
            body.pop("tools"); body.pop("tool_choice")
            body["output_config"] = {"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"label": {"type": "string", "enum": NAMES}}, "required": ["label"], "additionalProperties": False}}}
        if CL_TEMP[0] is not None and m == "claude": body["temperature"] = CL_TEMP[0]
        st, raw = post("api.anthropic.com", "/v1/messages", body, {"x-api-key": ENV["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        j = json.loads(raw)
        if st != 200: return {"raw": raw.decode("utf-8", "replace")[:300], "label": None, "loose": None, "tin": None, "tout": None, "status": st, "note": "error"}
        tu = [c for c in j.get("content", []) if c.get("type") == "tool_use"]
        if tu: inp = tu[0].get("input"); txt = json.dumps(inp, ensure_ascii=False)
        else:
            txt = "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text")
            try: inp = json.loads(txt)
            except Exception: inp = {}
        lab = inp.get("label") if isinstance(inp, dict) and inp.get("label") in LABELS else None; u = j.get("usage") or {}
        return {"raw": txt, "label": lab, "loose": lab, "tin": u.get("input_tokens"), "tout": u.get("output_tokens"), "status": st, "note": f"{j.get('model')} temp={CL_TEMP[0]}"}

GPT_EFFORT = ["none"]; GEM_THINK = [{"thinkingLevel": "minimal"}]; CL_TEMP = [None]

def timed(m, q):
    t = time.perf_counter()
    try:
        r = call(m, q); r["err"] = None if r.get("status") == 200 else f"HTTP {r.get('status')}"
    except Exception as e:
        r = {"raw": "", "label": None, "loose": None, "tin": None, "tout": None, "status": None, "note": "", "err": f"{type(e).__name__}: {e}"[:200]}
    r["sec"] = round(time.perf_counter() - t, 4); return r

def setup_console(cols=104, lines=26, size=22):
    os.system("")
    try:
        import ctypes
        from ctypes import wintypes
        class COORD(ctypes.Structure): _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]
        class FONT(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.ULONG), ("nFont", wintypes.DWORD), ("dwFontSize", COORD), ("FontFamily", wintypes.UINT),
                        ("FontWeight", wintypes.UINT), ("FaceName", wintypes.WCHAR * 32)]
        h = ctypes.windll.kernel32.GetStdHandle(-11); f = FONT(); f.cbSize = ctypes.sizeof(FONT)
        ctypes.windll.kernel32.GetCurrentConsoleFontEx(h, False, ctypes.byref(f)); f.dwFontSize.X = 0; f.dwFontSize.Y = size
        ctypes.windll.kernel32.SetCurrentConsoleFontEx(h, False, ctypes.byref(f))
    except Exception: pass
    os.system(f"mode con cols={cols} lines={lines}")
    try:
        import ctypes; ctypes.windll.kernel32.SetConsoleTitleW(TITLE)
        hwnd = ctypes.windll.kernel32.GetConsoleWindow(); ctypes.windll.user32.ShowWindow(hwnd, 9); ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception: pass
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

def start_rec(name):
    REC.mkdir(parents=True, exist_ok=True)
    try:
        import ctypes; ctypes.windll.kernel32.SetConsoleTitleW(TITLE); time.sleep(0.5)
    except Exception: pass
    out = REC / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{name}.mp4"
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "gdigrab", "-framerate", "30", "-draw_mouse", "0", "-i", f"title={TITLE}",
                          "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(0.8); REC_T0[0] = time.time() - 0.8
    return p, out, kst()
def stop_rec(rec, meta):
    p, out, t0 = rec; time.sleep(1.5)
    try: p.stdin.write(b"q"); p.stdin.flush(); p.wait(timeout=60)
    except Exception: p.kill()
    meta.update({"recorded_at_kst": t0, "ended_kst": kst(), "target": {"window": TITLE}, "tool": "ffmpeg gdigrab 30fps (bench.py 안에서 실험과 동시에 녹화)", "file": out.name})
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"); return out.name

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_sets():
    qf, af = HERE / "questions.jsonl", HERE / "answers.jsonl"
    if not af.exists():
        qf.write_text("".join(json.dumps({"id": i, "level": lv, "text": t}, ensure_ascii=False) + "\n" for i, lv, _, t in ITEMS), encoding="utf-8")
        af.write_text("".join(json.dumps({"id": i, "label": a}, ensure_ascii=False) + "\n" for i, _, a, _ in ITEMS), encoding="utf-8")
    return qf, af

def p90(xs): xs = sorted(xs); return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]

def summarize(rows):
    S = {}
    for m in list(MODEL) + list(BIG):
        rs = [r for r in rows if r["model"] == m]
        if not rs: continue
        secs = [r["sec"] for r in rs if not r["err"]]
        ok = sum(1 for r in rs if r["correct"]); loose = sum(1 for r in rs if r["loose_correct"])
        fmt = sum(1 for r in rs if not r["err"] and r["label"] is None); errs = sum(1 for r in rs if r["err"])
        tin = sum(r["tin"] or 0 for r in rs); tout = sum(r["tout"] or 0 for r in rs)
        pi, po = PRICE[m]; cost = tin / 1e6 * pi + tout / 1e6 * po
        lv = {}
        for L in ("쉬움", "중간", "어려움"):
            sub = [r for r in rs if r["level"] == L]; lv[L] = [sum(1 for r in sub if r["correct"]), len(sub)]
        cons = 0
        for qid in {r["qid"] for r in rs}:
            labs = [r["label"] for r in rs if r["qid"] == qid]
            if len(set(labs)) == 1 and len(labs) >= 2: cons += 1
        S[m] = {"model_id": mid(m), "served": sorted({r["note"] for r in rs if r["note"]})[:3], "n": len(rs), "correct": ok, "loose_correct": loose,
                "format_errors": fmt, "call_errors": errs, "by_level": lv, "consistent_3of3": cons,
                "median_sec": round(statistics.median(secs), 4) if secs else None, "p90_sec": round(p90(secs), 4) if secs else None,
                "mean_sec": round(statistics.mean(secs), 4) if secs else None, "min_sec": round(min(secs), 4) if secs else None, "max_sec": round(max(secs), 4) if secs else None,
                "tokens_in": tin, "tokens_out": tout, "cost_usd": round(cost, 6), "price_per_mtok": PRICE[m]}
    if "jev" in S:
        for m in S:
            if m != "jev" and S[m]["median_sec"] and S["jev"]["median_sec"]:
                S[m]["x_slower_than_jev_median"] = round(S[m]["median_sec"] / S["jev"]["median_sec"], 2)
                S[m]["x_costlier_than_jev"] = round(S[m]["cost_usd"] / S["jev"]["cost_usd"], 1) if S["jev"]["cost_usd"] else None
    return S

def run(record):
    setup_console(); OUT.mkdir(exist_ok=True)
    qf, af = write_sets(); h0 = sha(af)
    rec = start_rec("jevbench") if record else None
    print(f"{C['b']}{C['w']}고객 문의 30개 분류 · 4개 모델 · 3회차  ({kst()} KST, 같은 PC·같은 회선, 문항마다 호출 1번){C['0']}")
    print(f"{C['d']}정답 파일 SHA-256  {h0[:16]}…{h0[-8:]}{C['0']}")
    for m in MODEL: print(f"  {C[COL[m]]}{SHOW[m]:<17}{C['0']} {MODEL[m]}  ·  100만 토큰당 입력 {PRICE[m][0]}달러 / 출력 {PRICE[m][1]}달러")
    time.sleep(3)
    print(f"{C['d']}워밍업(결과에서 제외) …{C['0']}")
    for m in MODEL:
        w = timed(m, "테스트 문의입니다. 배송 언제 와요?"); print(f"{C['d']}  {SHOW[m]:<17} {w['sec']:.3f}초 {w['label']} {w['err'] or ''}{C['0']}")
    rows = []; orders = [["jev", "gpt", "gemini", "claude"], ["claude", "gemini", "gpt", "jev"], ["gpt", "jev", "claude", "gemini"]]
    for rnd in (1, 2, 3):
        qs = list(ITEMS); random.Random(20261102 + rnd).shuffle(qs)
        for m in orders[rnd - 1]:
            print(f"\n{C['b']}{C[COL[m]]}── {rnd}회차 · {SHOW[m]} ({MODEL[m]}) ──{C['0']}"); t_lab = roff()
            for k, (qid, lv, ans, text) in enumerate(qs, 1):
                r = timed(m, text); okk = r["label"] == ans
                mark = f"{C['g']}✓{C['0']}" if okk else f"{C['r']}✗{C['0']}"
                got = r["label"] or (f"형식오류" if not r["err"] else "호출실패")
                print(f"{C[COL[m]]}{SHOW[m]:<16}{C['0']} {k:>2}/30 {qid} [{lv}] 정답 {ans:<6} → {got:<6} {mark}  {C['w']}{r['sec']:.3f}초{C['0']}" + (f" {C['r']}{r['err']}{C['0']}" if r["err"] else ""))
                rows.append({"round": rnd, "model": m, "order_in_round": orders[rnd - 1].index(m) + 1, "k": k, "qid": qid, "level": lv, "answer": ans, "label": r["label"],
                             "loose_label": r["loose"], "correct": okk, "loose_correct": r["loose"] == ans, "sec": r["sec"], "tin": r["tin"], "tout": r["tout"],
                             "status": r["status"], "err": r["err"], "raw": r["raw"], "note": r["note"], "prob": r.get("prob"), "rec_t": roff(), "at_kst": kst()})
            rows[-1]["block_rec_start"] = t_lab
    S = summarize(rows); h1 = sha(af)
    print(f"\n{C['b']}{C['w']}결과 (90번 = 30문항 × 3회차){C['0']}")
    for m in MODEL:
        s = S[m]
        print(f"  {C[COL[m]]}{SHOW[m]:<17}{C['0']} 정답 {s['correct']:>2}/90  형식오류 {s['format_errors']}  실패 {s['call_errors']}  중앙값 {s['median_sec']:.3f}초  p90 {s['p90_sec']:.3f}초  비용 {s['cost_usd']:.5f}달러")
    print(f"{C['d']}정답 파일 SHA-256  {h1[:16]}…{h1[-8:]}  (실험 전과 {'같음' if h0 == h1 else '다름!'}){C['0']}")
    meta = {"sha256_before": h0, "sha256_after": h1, "started": rows[0]["at_kst"], "ended": kst(), "gpt_effort": GPT_EFFORT[0], "gemini_thinking": GEM_THINK[0], "claude_temp": CL_TEMP[0]}
    if rec: meta["recording"] = stop_rec(rec, {"what": "bench.py --run 전체"})
    (OUT / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with (OUT / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + (["block_rec_start"] if "block_rec_start" not in rows[0] else []), extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps({"meta": meta, "summary": S}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{C['b']}끝 · {kst()}{C['0']}"); time.sleep(5)

def run_big(record):
    setup_console(); OUT.mkdir(exist_ok=True)
    qf, af = write_sets(); h0 = sha(af)
    rec = start_rec("jevbench_big") if record else None
    print(f"{C['b']}{C['w']}추가 비교: 큰 모델 2개 (출처가 GPT·Fable과 비교) · 같은 30문항 1회차 · Jev도 같은 시간에 다시 ({kst()} KST){C['0']}")
    print(f"{C['d']}정답 파일 SHA-256  {h0[:16]}…{h0[-8:]}  · 큰 모델은 추론 설정 기본값{C['0']}")
    for m in ("jev", "sol", "fable"): print(f"  {C[COL[m]]}{SHOW[m]:<17}{C['0']} {mid(m)}  ·  100만 토큰당 입력 {PRICE[m][0]}달러 / 출력 {PRICE[m][1]}달러")
    time.sleep(3); print(f"{C['d']}워밍업(결과에서 제외) …{C['0']}")
    for m in ("sol", "fable", "jev"):
        w = timed(m, "테스트 문의입니다. 배송 언제 와요?"); print(f"{C['d']}  {SHOW[m]:<17} {w['sec']:.3f}초 {w['label']} {w['err'] or ''}{C['0']}")
    rows = []; qs = list(ITEMS); random.Random(20261102 + 1).shuffle(qs)
    for m in ("sol", "jev", "fable"):
        print(f"\n{C['b']}{C[COL[m]]}── 추가 · {SHOW[m]} ({mid(m)}) ──{C['0']}")
        for k, (qid, lv, ans, text) in enumerate(qs, 1):
            r = timed(m, text); okk = r["label"] == ans
            mark = f"{C['g']}✓{C['0']}" if okk else f"{C['r']}✗{C['0']}"
            got = r["label"] or ("형식오류" if not r["err"] else "호출실패")
            print(f"{C[COL[m]]}{SHOW[m]:<16}{C['0']} {k:>2}/30 {qid} [{lv}] 정답 {ans:<6} → {got:<6} {mark}  {C['w']}{r['sec']:.3f}초{C['0']}" + (f" {C['r']}{r['err']}{C['0']}" if r["err"] else ""))
            rows.append({"round": "big", "model": m, "k": k, "qid": qid, "level": lv, "answer": ans, "label": r["label"], "loose_label": r["loose"], "correct": okk,
                         "loose_correct": r["loose"] == ans, "sec": r["sec"], "tin": r["tin"], "tout": r["tout"], "status": r["status"], "err": r["err"], "raw": r["raw"],
                         "note": r["note"], "prob": r.get("prob"), "rec_t": roff(), "at_kst": kst()})
    S = summarize(rows); h1 = sha(af)
    print(f"\n{C['b']}{C['w']}추가 비교 결과 (30번씩){C['0']}")
    for m in ("jev", "sol", "fable"):
        s = S[m]; print(f"  {C[COL[m]]}{SHOW[m]:<17}{C['0']} 정답 {s['correct']:>2}/30  중앙값 {s['median_sec']:.3f}초  p90 {s['p90_sec']:.3f}초  비용 {s['cost_usd']:.5f}달러")
    print(f"{C['d']}정답 파일 SHA-256  {h1[:16]}…{h1[-8:]}  (실험 전과 {'같음' if h0 == h1 else '다름!'}){C['0']}")
    meta = {"sha256_before": h0, "sha256_after": h1, "started": rows[0]["at_kst"], "ended": kst(), "big_settings": "추론·사고 설정 기본값(지정 안 함)"}
    if rec: meta["recording"] = stop_rec(rec, {"what": "bench.py --big 전체"})
    (OUT / "results_big.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT / "summary_big.json").write_text(json.dumps({"meta": meta, "summary": S}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{C['b']}끝 · {kst()}{C['0']}"); time.sleep(5)

def probe():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    for m in MODEL:
        r = timed(m, ITEMS[0][3]); print(m, r["sec"], r["label"], r["err"], r["note"], (r["raw"] or "")[:200], r["tin"], r["tout"])

if __name__ == "__main__":
    a = sys.argv[1:]
    for x in a:
        if x.startswith("--gpt-effort="): GPT_EFFORT[0] = x.split("=", 1)[1]
        if x.startswith("--gem-think="): GEM_THINK[0] = json.loads(x.split("=", 1)[1])
        if x == "--claude-notemp": CL_TEMP[0] = None
    if "--probe" in a: probe()
    elif "--run" in a: run("--record" in a)
    elif "--big" in a: run_big("--record" in a)
    elif "--probe-big" in a:
        for m in ("sol", "fable"):
            r = timed(m, ITEMS[0][3]); print(m, r["sec"], r["label"], r["err"], r["note"], (r["raw"] or "")[:200], r["tin"], r["tout"])






