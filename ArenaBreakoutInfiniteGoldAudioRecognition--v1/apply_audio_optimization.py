# -*- coding: utf-8 -*-
"""Apply low-risk template-cache optimizations to main.py.

Usage:
  1. Back up main.py.
  2. Put this script beside main.py.
  3. Run: python apply_audio_optimization.py
"""
from pathlib import Path
import time

MAIN = Path(__file__).resolve().with_name("main.py")


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Cannot safely apply {label!r}: expected one matching block, found {count}. "
            "No file was written; main.py may differ from the supplied version."
        )
    return text.replace(old, new, 1)


def main():
    if not MAIN.exists():
        print(f"ERROR: {MAIN} not found. Put this script next to main.py.")
        return 1

    original = MAIN.read_text(encoding="utf-8")
    text = original

    text = replace_once(
        text,
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, field\n",
        "dataclass field import",
    )

    text = replace_once(
        text,
        """@dataclass
class Template:
    name: str
    payload: Any
    feature: np.ndarray
    duration: float
""",
        """@dataclass
class Template:
    name: str
    payload: Any
    feature: np.ndarray
    duration: float
    # Precomputed time-scaled feature matrices, keyed by scale.
    scaled_features: dict[float, np.ndarray] = field(default_factory=dict)
""",
        "Template cache field",
    )

    text = replace_once(
        text,
        """                templates.append(
                    Template(
                        name=filename,
                        payload=payload,
                        feature=feature,
                        duration=duration,
                    )
                )
""",
        """                # Time scaling depends only on the static template, so do it once.
                scaled_features = {
                    float(scale): self.scale_template(
                        feature,
                        float(scale),
                    )
                    for scale in self.time_scales
                }

                templates.append(
                    Template(
                        name=filename,
                        payload=payload,
                        feature=feature,
                        duration=duration,
                        scaled_features=scaled_features,
                    )
                )
""",
        "precompute scaled templates",
    )

    start_marker = "    def sliding_similarity(\n"
    end_marker = "    # --------------------------------------------------------\n    # 搜索回环设备\n"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError(
            "Cannot safely locate sliding_similarity() block. No file was written."
        )

    new_method = """    def sliding_similarity(
        self,
        live: np.ndarray,
        template: Template | np.ndarray,
    ) -> tuple[float, int]:

        \"""
        在实时特征窗口中滑动搜索模板。

        传入 Template 时复用启动阶段预计算的多尺度特征；
        传入 ndarray 时保留兼容行为。
        返回：最佳相似度、最佳位置对应的特征帧索引。
        \"""

        if isinstance(template, Template):
            base_feature = template.feature
            cached_scaled = template.scaled_features
        else:
            base_feature = template
            cached_scaled = {}

        if (
            live.shape[0] != base_feature.shape[0]
            or live.shape[1] < 3
            or base_feature.shape[1] < 3
        ):
            return -1.0, 0

        live_length = live.shape[1]
        best_score = -1.0
        best_offset = 0

        for scale in self.time_scales:
            scale_key = float(scale)
            target = cached_scaled.get(scale_key)

            # 兼容直接传入 ndarray 的旧调用方式。
            if target is None:
                target = self.scale_template(
                    base_feature,
                    scale_key,
                )

            target_length = target.shape[1]
            if target_length > live_length:
                continue

            windows = np.lib.stride_tricks.sliding_window_view(
                live,
                window_shape=target_length,
                axis=1,
            )

            scores = np.einsum(
                "dkl,dl->k",
                windows,
                target,
                optimize=True,
            )
            scores /= target_length

            max_score = float(np.max(scores))

            # 在近似并列的位置中优先选择较新的匹配位置。
            possible = np.flatnonzero(
                scores >= max_score - 0.005
            )
            offset = (
                int(possible[-1])
                if len(possible)
                else int(np.argmax(scores))
            )

            if max_score > best_score:
                best_score = max_score
                best_offset = offset
            elif abs(max_score - best_score) <= 0.005:
                if offset > best_offset:
                    best_offset = offset

        return (
            float(np.clip(best_score, -1.0, 1.0)),
            best_offset,
        )

"""
    text = text[:start] + new_method + text[end:]

    text = replace_once(
        text,
        """                    score, offset = self.sliding_similarity(
                        live_feature,
                        template.feature,
                    )
""",
        """                    score, offset = self.sliding_similarity(
                        live_feature,
                        template,
                    )
""",
        "use cached template in matching loop",
    )

    text = replace_once(
        text,
        """                candidates = []

                for template in templates:
""",
        """                match_started = time.perf_counter()
                candidates = []

                for template in templates:
""",
        "start match timer",
    )

    text = replace_once(
        text,
        """                candidates.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                if not candidates:
""",
        """                candidates.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )
                match_ms = (time.perf_counter() - match_started) * 1000.0

                if not candidates:
""",
        "finish match timer",
    )

    text = replace_once(
        text,
        """                    self.log(
                        f"[候选] RMS={rms:.4f} | "
                        f"{preview} | "
                        f"分差={margin:.3f}"
                    )
""",
        """                    self.log(
                        f"[候选] RMS={rms:.4f} | "
                        f"{preview} | "
                        f"分差={margin:.3f} | "
                        f"匹配耗时={match_ms:.1f}ms"
                    )
""",
        "log matching duration",
    )

    backup = MAIN.with_suffix(".py.bak")
    if backup.exists():
        backup = MAIN.with_name(f"main.py.{int(time.time())}.bak")
    backup.write_text(original, encoding="utf-8")
    MAIN.write_text(text, encoding="utf-8")

    print("Optimization applied successfully.")
    print(f"Backup: {backup}")
    print(f"Updated: {MAIN}")
    print("Feature formulas and thresholds were not changed.")
    print("Compare CPU usage, matching time, false positives and false negatives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
