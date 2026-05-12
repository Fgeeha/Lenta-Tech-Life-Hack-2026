"""Дедупликация: один track_id → одна строка CSV (заглушка)."""

from shelf.schema import PriceTag


def dedup_by_track(tagged: list[tuple[int, PriceTag]]) -> list[PriceTag]:
    """Из списка (track_id, PriceTag) выбрать лучший кадр для каждого трека.

    «Лучший» — с наибольшей площадью бокса.
    """
    best: dict[int, PriceTag] = {}
    for track_id, tag in tagged:
        area = (tag.x_max - tag.x_min) * (tag.y_max - tag.y_min)
        prev = best.get(track_id)
        if prev is None:
            best[track_id] = tag
        else:
            prev_area = (prev.x_max - prev.x_min) * (prev.y_max - prev.y_min)
            if area > prev_area:
                best[track_id] = tag
    return list(best.values())
