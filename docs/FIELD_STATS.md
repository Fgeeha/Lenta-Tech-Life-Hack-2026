# GT Field Value Distribution

Computed from all 5 labeled GT videos (274 price tags total).
Columns: % where GT is empty/nan (auto-match regardless of prediction),
         % where GT = "нет" (auto-match if we output "нет"),
         % where GT has a real value (must be recognized correctly to contribute).

## Fields in EVAL_FIELDS_FULL (metric@80% denominator, 11 fields)

| Field           | empty% | absent% ("нет") | real% | Action for metric@80% |
|-----------------|--------|-----------------|-------|------------------------|
| price_discount  |   0.0% |           100.0%|   0.0%| ✅ Auto-match always; we already output "нет" at 97.1% accuracy |
| product_name    |   0.0% |             0.0%|  100.0%| ❌ Must OCR Cyrillic name correctly — 0.7% current accuracy |
| price_default   |   0.0% |             0.4%|  99.6%| ❌ Must OCR regular price — 5.8% current accuracy |
| price_card      |   0.0% |             1.1%|  98.9%| ⚠️  42.7% current accuracy (best OCR field after price_discount) |
| barcode         |   0.7% |             1.1%|  98.2%| ❌ Must decode EAN-13 — 2.9% current; linear barcode decode bottleneck |
| discount_amount |   0.4% |             5.1%|  94.5%| ⚠️  21.9% current; improves when both prices are correct |
| id_sku          |   4.4% |             1.1%|  94.5%| ❌ Must extract 12-digit SKU — 5.1% current; 0% fill rate → must come from catalog |
| qr_code_barcode |   1.8% |             0.0%|  98.2%| ❌ Must decode QR code barcode field — 2.9% current; QR decode bottleneck |
| price1_qr       |   1.8% |             0.0%|  98.2%| ❌ Must read regular price from QR URL — 7.3% current (filled from price_default OCR) |
| price2_qr       |   2.9% |             8.8%|  88.3%| ⚠️  16.8% current (derived as price1_qr × 0.95) |
| price4_qr       |   1.8% |             5.1%|  93.1%| ⚠️  46.7% current (filled from price_card OCR) |

## Key Insights

### Why metric@80% is stuck at 0.011 (3/274)

For a tag to pass 9/11 correct:
- `price_discount` is a free field (100% auto-match).
- The remaining 10 fields all require **real recognition** in GT 88-100% of rows.
- A typical tag currently gets ~6/11 correct:
  `price_discount` ✓ + `price_card` (43%) + `price4_qr` (47%, correlated) +
  `discount_amount` (22%) + `price2_qr` (17%) + `price1_qr` (7%).
- The 3 passing tags in video 26_12-20 had **linear barcode decoded**, which cascaded:
  `barcode` ✓ → `qr_code_barcode` ✓ (sync) → `product_name` ✓ (catalog) →
  enough additional fields to reach 9/11.

### What would move the needle most

Rank by expected metric@80% gain per unit of implementation effort:

1. **QR code decode** (WeChatQR integration): Each decoded QR gives 4 fields at once —
   `qr_code_barcode`, `price1_qr`, `price2_qr` (URL or derived), `price4_qr` (from URL).
   Currently 0 QR decodes. WeChatQR has much higher success rate on degraded codes.

2. **Linear barcode decode** (OpenCV BarcodeDetector + better preprocessing):
   Each decoded barcode enables `barcode` + `qr_code_barcode` sync + catalog lookup
   (`product_name` + `id_sku`). Currently 4/274 decoded.

3. **id_sku from catalog** (fix now done): Once barcode is decoded → catalog lookup →
   add `id_sku` fill. Adds 1 more field per barcode-decoded tag for free.

4. **price_default OCR improvement**: Currently 5.8%. Better ROI isolation or OCR model
   would help price_default + price1_qr (via fill) + discount_amount (via derivation).

5. **product_name OCR** (TrOCR fine-tune on synthetic data — Stage E): Currently 0.7%.
   Biggest lift but requires model training.

## Fields NOT in metric@80% denominator (non-evaluated fields)

These fields are present in the CSV but not counted toward metric@80%:
`filename`, `print_datetime`, `code`, `additional_info`, `color`, `special_symbols`,
`frame_timestamp`, `x_min`, `y_min`, `x_max`, `y_max`, `price3_qr`,
`wholesale_level_1_count`, `wholesale_level_1_price`, `wholesale_level_2_count`,
`wholesale_level_2_price`, `action_price_qr`, `action_code_qr`.

In GT: almost all of these have 0% real values (always empty/absent).
Setting them all to "нет" or empty has **no impact on metric@80%**.
