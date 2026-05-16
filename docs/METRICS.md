# История метрики

Метрика: доля ценников, у которых ≥80% полей распознаны верно.

| Дата | Commit | Видео | Метрика | Примечание |
|------|--------|-------|---------|------------|
| —    | —      | —     | —       | таблица пустая до этапа 8 |
| 2026-05-12 | 3eda52f | 43_15 | 0.000 | matched=23/29 avg_field=0.09 |
| 2026-05-13 | 85155e1 | pseudo-labels | — | train=121boxes/24frames val=24boxes/2frames hq=0 |
| 2026-05-13 | 85155e1 | YOLO-train | — | mAP50=0.368 mAP50-95=0.100 epochs=50 batch=8 |
| 2026-05-13 | 85155e1 | 25_12-20 | 0.000 | matched=0/57 avg_field=0.00 |
| 2026-05-13 | 85155e1 | 25_12-20 | 0.000 | matched=57/57 avg_field=0.13 |
| 2026-05-13 | 85155e1 | 26_12-20 | 0.000 | matched=71/71 avg_field=0.10 |
| 2026-05-13 | 85155e1 | 43_15 | 0.000 | matched=29/29 avg_field=0.10 |
| 2026-05-13 | 574d2f4 | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 574d2f4 | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 3c207bf | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 3c207bf | 43_15 | 0.000 | matched=2/29 avg_field=0.09 |
| 2026-05-13 | 3c207bf | 43_15 | 0.000 | matched=29/29 avg_field=0.15 |
| 2026-05-13 | 3c207bf | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 3c207bf | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 3c207bf | 25_12-20 | 0.000 | matched=55/57 avg_field=0.14 |
| 2026-05-13 | 3c207bf | 26_12-20 | 0.000 | matched=71/71 avg_field=0.14 |
| 2026-05-13 | 3c207bf | 43_15 | 0.000 | matched=29/29 avg_field=0.17 |
| 2026-05-13 | 459ec0d | ceiling (GT bboxes) | 0.006 | 25_12-20=0.000  26_12-20=0.014  43_15=0.000 |
| 2026-05-13 | 459ec0d | 25_12-20 | 0.000 | matched=55/57 avg_field=0.16 |
| 2026-05-13 | 459ec0d | 26_12-20 | 0.028 | matched=71/71 avg_field=0.16 |
| 2026-05-13 | 459ec0d | 43_15 | 0.000 | matched=29/29 avg_field=0.24 |
| 2026-05-13 | 432c971 | 25_12-20 | 0.000 | matched=57/57 avg_field=0.20 |
| 2026-05-13 | 432c971 | 26_12-20 | 0.028 | matched=71/71 avg_field=0.18 |
| 2026-05-13 | 432c971 | 43_15 | 0.000 | matched=29/29 avg_field=0.20 |
| 2026-05-13 | 2bf2154 | ceiling (GT bboxes) | 0.006 | 25_12-20=0.000  26_12-20=0.014  43_15=0.000 |
| 2026-05-14 | 3029815 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | 3029815 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | 3029815 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | d5ada29 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | d5ada29 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | 7221017 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | f687799 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | f687799 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-15 | d40a4cd | ceiling (GT bboxes) | 0.011 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-15 | bd06c60 | ceiling (GT bboxes) | 0.011 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-15 | 3838a6d | ceiling (GT bboxes) | 0.011 | avg_field=0.222; no_qr=0.004; avg_no_qr=0.251; has_qr_no_qr=0.004; no_qr_gt_no_qr=0.000; bc=4; qr_bc=4; 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-15 | a8f65cf | ceiling (GT bboxes) | 0.011 | avg_field=0.226; no_qr=0.004; avg_no_qr=0.251; has_qr_no_qr=0.004; no_qr_gt_no_qr=0.000; bc=4; qr_bc=4; 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-16 | 5ef5738 | ceiling (GT bboxes) | 0.011 | avg_field=0.227; no_qr=0.004; avg_no_qr=0.252; has_qr_no_qr=0.004; no_qr_gt_no_qr=0.000; bc=4; qr_bc=4; 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-16 | fbdad91 | ceiling (GT bboxes) | 0.011 | avg_field=0.227; no_qr=0.004; avg_no_qr=0.252; has_qr_no_qr=0.004; no_qr_gt_no_qr=0.000; bc=4; qr_bc=4; 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-16 | eae2032 | ceiling (GT bboxes) | 0.011 | avg_field=0.227; no_qr=0.004; avg_no_qr=0.252; has_qr_no_qr=0.004; no_qr_gt_no_qr=0.000; bc=4; qr_bc=4; 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-16 | 46767be | ceiling (GT bboxes) | 0.011 | avg_field=0.227; no_qr=0.004; avg_no_qr=0.252; has_qr_no_qr=0.004; no_qr_gt_no_qr=0.000; bc=4; qr_bc=4; 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-16 | 4be75bc | ceiling (GT bboxes) | 0.011 | Stage A final: avg_field=0.227; GT analysis: 98.2% tags need QR; barcode zone unreadable at 8x zoom; 3/274 QR decoded |
| 2026-05-16 | 7731bf5 | production pipeline | — | Stage B: QR-zone sharpness tracking in ByteTrack; fallback decode_qr on best_qr_frame; production eval running |
| 2026-05-16 | 14b5bc0 | ceiling+multiframe | 0.011 | ±300ms 50ms steps: NO additional QR decodes. 271/274 QR codes unreadable in ALL video frames. Fundamental video quality limit confirmed. 3/274 QR only at ts=15833ms in 26_12-20. |
| 2026-05-16 | 1cb2096 | labeled-5 fast(k=1,1fps) | 0.000 | avg_field=0.101; qr_bc=0; 1fps sampling MISSED ts=15833ms (QR window ~150ms wide). Adaptive 4fps required to catch QR-readable frames. |
