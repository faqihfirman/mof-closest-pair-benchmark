"""Dataset 2D tetap (P1..P7) + tiga algoritma closest-pair yang diinstrumentasi.

Dipakai oleh tampilan "Demo 2D" di dashboard: tiap fungsi menjalankan algoritmanya sambil
mencatat log operasi (`log`), supaya UI bisa memutar ulang langkah demi langkah tanpa
langkah di-hardcode secara manual. Ganti `POINTS` -> visualisasi tetap otomatis benar.
"""
from __future__ import annotations

import math
from typing import TypedDict

Pair = tuple[str, str]

POINTS: dict[str, tuple[float, float]] = {
    "P1": (1, 1), "P2": (2, 5), "P3": (4, 3), "P4": (5, 4),
    "P5": (7, 1), "P6": (8, 6), "P7": (9, 2),
}


class LogEntry(TypedDict, total=False):
    type: str                 # compare | update | skip | divide | strip | base_case | done
    points: list[str]
    distance: float
    currentBest: dict         # {"dist": float, "pair": [str, str]}
    note: str
    highlight: dict           # divider / strip / activeGroup / left / right (opsional)
    meta: dict                # info tambahan khusus rendering (current, remaining, ...)


class AlgoResult(TypedDict):
    log: list[LogEntry]
    finalDistance: float
    finalPair: Pair
    comparisonCount: int


def _dist(a: str, b: str) -> float:
    (x1, y1), (x2, y2) = POINTS[a], POINTS[b]
    return math.hypot(x1 - x2, y1 - y2)


def _best_dict(dist: float, pair: Pair) -> dict:
    return {"dist": dist, "pair": list(pair)}


def state_at(log: list[LogEntry], step_idx: int) -> dict:
    """Ringkasan visual di langkah `step_idx`: gabungan entri saat ini + info yang masih
    berlaku dari entri sebelumnya (divider/strip/grup tidak diulang tiap langkah di log).
    """
    entry = log[step_idx]
    best = entry.get("currentBest")
    divider = strip = left = right = group = current = remaining = None
    for past in log[step_idx::-1]:
        highlight = past.get("highlight") or {}
        meta = past.get("meta") or {}
        if divider is None and "divider" in highlight:
            divider = highlight["divider"]
        if strip is None and "strip" in highlight:
            strip = highlight["strip"]
        if left is None and "left" in highlight:
            left, right = highlight["left"], highlight["right"]
        if group is None and "activeGroup" in highlight:
            group = highlight["activeGroup"]
        if current is None and "current" in meta:
            current, remaining = meta["current"], meta["remaining"]
        if best is None and past.get("currentBest") is not None:
            best = past["currentBest"]
        if divider is not None and strip is not None and current is not None and best is not None:
            break
    best_pair = tuple(best["pair"]) if best and best["pair"][0] else None
    return {
        "entry": entry,
        "best_pair": best_pair,
        "best_dist": best["dist"] if best else math.inf,
        "compare_pair": entry["points"] if entry["type"] == "compare" else None,
        "skip_points": entry["points"] if entry["type"] == "skip" else None,
        "divider": divider, "strip": strip, "left": left, "right": right, "group": group,
        "current": current, "remaining": remaining,
    }


# ---------------------------------------------------------------- 1. Naive
def run_naive(points: dict[str, tuple[float, float]] = POINTS) -> AlgoResult:
    """Dua loop bersarang, urutan sesuai urutan dict `points`."""
    ids = list(points)
    log: list[LogEntry] = []
    best_dist = math.inf
    best_pair: Pair = ("", "")
    comparisons = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            d = _dist(a, b)
            comparisons += 1
            log.append({"type": "compare", "points": [a, b], "distance": d,
                        "currentBest": _best_dict(min(best_dist, d), best_pair if d >= best_dist else (a, b)),
                        "note": f"bandingkan {a}-{b}: jarak {d:.3f}"})
            if d < best_dist:
                best_dist, best_pair = d, (a, b)
                log.append({"type": "update", "points": [a, b], "distance": d,
                            "currentBest": _best_dict(best_dist, best_pair),
                            "note": f"jarak lebih kecil, update terbaik ke {a}-{b} ({d:.3f})"})
    log.append({"type": "done", "points": list(best_pair), "distance": best_dist,
                "currentBest": _best_dict(best_dist, best_pair),
                "note": f"selesai: pasangan terdekat {best_pair[0]}-{best_pair[1]} ({best_dist:.3f})"})
    return {"log": log, "finalDistance": best_dist, "finalPair": best_pair, "comparisonCount": comparisons}


# ---------------------------------------------------------------- 2. Greedy (Algo. 2 paper)
def run_greedy(start_id: str, points: dict[str, tuple[float, float]] = POINTS) -> AlgoResult:
    """Algo. 2: dari `start_id`, selalu lompat ke titik tersisa terdekat dalam radius
    `shortest_dis` saat ini; berhenti kalau tidak ada kandidat dalam radius itu (bukan bug,
    ini kondisi berhenti yang didefinisikan paper -> penyebab greedy bisa salah).
    """
    remaining = [p for p in points if p != start_id]
    current = start_id
    shortest_dis = math.inf
    best_pair: Pair = ("", "")
    log: list[LogEntry] = []
    comparisons = 0

    while remaining:
        candidates = []
        for other in remaining:
            d = _dist(current, other)
            comparisons += 1
            in_radius = d < shortest_dis
            candidates.append((other, d))
            log.append({"type": "compare", "points": [current, other], "distance": d,
                        "currentBest": _best_dict(shortest_dis if shortest_dis != math.inf else d,
                                                  best_pair if best_pair != ("", "") else (current, other)),
                        "note": f"cek {other} dari {current}: jarak {d:.3f}" +
                                ("" if in_radius else " (di luar radius saat ini)"),
                        "highlight": {"activeGroup": [current, other]},
                        "meta": {"current": current, "remaining": list(remaining)}})
        in_radius_candidates = [(o, d) for o, d in candidates if d < shortest_dis]
        if not in_radius_candidates:
            log.append({"type": "skip", "points": [current], "currentBest": _best_dict(shortest_dis, best_pair),
                        "note": "tidak ada titik dalam radius, algoritma berhenti di sini",
                        "meta": {"current": current, "remaining": list(remaining)}})
            break

        nearest, nearest_dist = min(in_radius_candidates, key=lambda item: item[1])
        if nearest_dist < shortest_dis:
            shortest_dis, best_pair = nearest_dist, (current, nearest)
            log.append({"type": "update", "points": [current, nearest], "distance": nearest_dist,
                        "currentBest": _best_dict(shortest_dis, best_pair),
                        "note": f"jarak lebih kecil, update terbaik ke {current}-{nearest} ({nearest_dist:.3f})",
                        "meta": {"current": current, "remaining": list(remaining)}})
        current = nearest
        remaining.remove(nearest)

    log.append({"type": "done", "points": list(best_pair), "distance": shortest_dis,
                "currentBest": _best_dict(shortest_dis, best_pair),
                "note": f"selesai: pasangan terdekat {best_pair[0]}-{best_pair[1]} ({shortest_dis:.3f})"})
    return {"log": log, "finalDistance": shortest_dis, "finalPair": best_pair, "comparisonCount": comparisons}


# ---------------------------------------------------------------- 3. Divide & Conquer
def run_dnc(points: dict[str, tuple[float, float]] = POINTS) -> AlgoResult:
    """Urutkan berdasarkan x, bagi dua di tengah, base case n<=3 brute force, gabungkan
    dengan strip di sekitar garis pembagi (disaring lagi berdasarkan y).
    """
    log: list[LogEntry] = []
    comparisons = 0

    def brute(ids: list[str]) -> tuple[float, Pair]:
        nonlocal comparisons
        best_dist, best_pair = math.inf, ("", "")
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                d = _dist(a, b)
                comparisons += 1
                log.append({"type": "compare", "points": [a, b], "distance": d,
                            "currentBest": _best_dict(min(best_dist, d), best_pair if d >= best_dist else (a, b)),
                            "note": f"base case: bandingkan {a}-{b} ({d:.3f})",
                            "highlight": {"activeGroup": ids}})
                if d < best_dist:
                    best_dist, best_pair = d, (a, b)
                    log.append({"type": "update", "points": [a, b], "distance": d,
                                "currentBest": _best_dict(best_dist, best_pair),
                                "note": f"update terbaik ke {a}-{b} ({d:.3f})"})
        return best_dist, best_pair

    def solve(ids: list[str]) -> tuple[float, Pair]:
        nonlocal comparisons
        if len(ids) <= 3:
            log.append({"type": "base_case", "points": ids, "note": f"kelompok kecil ({len(ids)} titik): brute force",
                        "currentBest": _best_dict(math.inf, ("", "")), "highlight": {"activeGroup": ids}})
            return brute(ids)

        mid = len(ids) // 2
        left_ids, right_ids = ids[:mid], ids[mid:]
        x_mid = (POINTS[ids[mid - 1]][0] + POINTS[ids[mid]][0]) / 2
        log.append({"type": "divide", "points": ids, "note": f"bagi di x={x_mid:.3f}: kiri {left_ids}, kanan {right_ids}",
                    "currentBest": _best_dict(math.inf, ("", "")),
                    "highlight": {"divider": x_mid, "left": left_ids, "right": right_ids}})

        d_left, pair_left = solve(left_ids)
        d_right, pair_right = solve(right_ids)
        d, best_pair = (d_left, pair_left) if d_left <= d_right else (d_right, pair_right)

        strip_ids = [p for p in ids if abs(POINTS[p][0] - x_mid) < d]
        strip_lo = x_mid - d
        strip_hi = x_mid + d
        log.append({"type": "strip", "points": strip_ids,
                    "note": f"pita |x-{x_mid:.3f}| < {d:.3f}: {len(strip_ids)} titik diperiksa lagi",
                    "currentBest": _best_dict(d, best_pair),
                    "highlight": {"divider": x_mid, "strip": [strip_lo, strip_hi], "activeGroup": strip_ids}})
        strip_by_y = sorted(strip_ids, key=lambda p: POINTS[p][1])
        for i in range(len(strip_by_y)):
            for j in range(i + 1, len(strip_by_y)):
                a, b = strip_by_y[i], strip_by_y[j]
                if POINTS[b][1] - POINTS[a][1] >= d:
                    break
                dd = _dist(a, b)
                comparisons += 1
                log.append({"type": "compare", "points": [a, b], "distance": dd,
                            "currentBest": _best_dict(min(d, dd), best_pair if dd >= d else (a, b)),
                            "note": f"bandingkan di strip {a}-{b} ({dd:.3f})",
                            "highlight": {"divider": x_mid, "strip": [strip_lo, strip_hi]}})
                if dd < d:
                    d, best_pair = dd, (a, b)
                    log.append({"type": "update", "points": [a, b], "distance": dd,
                                "currentBest": _best_dict(d, best_pair),
                                "note": f"strip memperbaiki jawaban ke {a}-{b} ({dd:.3f})"})
        return d, best_pair

    ids_by_x = sorted(points, key=lambda p: points[p][0])
    best_dist, best_pair = solve(ids_by_x)
    log.append({"type": "done", "points": list(best_pair), "distance": best_dist,
                "currentBest": _best_dict(best_dist, best_pair),
                "note": f"selesai: pasangan terdekat {best_pair[0]}-{best_pair[1]} ({best_dist:.3f})"})
    return {"log": log, "finalDistance": best_dist, "finalPair": best_pair, "comparisonCount": comparisons}
