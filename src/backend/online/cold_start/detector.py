from __future__ import annotations


def is_cold_start(scene: str, user_idx: int, user_requests: dict[int, list[int]], request_threshold: int) -> bool:
    if scene != "rec":
        return False
    reqs = user_requests.get(int(user_idx), [])
    return len(reqs) < int(request_threshold)
