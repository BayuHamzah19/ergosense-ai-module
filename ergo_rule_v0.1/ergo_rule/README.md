# ergo_rule v0.1: mesin aturan EAR ERGO-SENSE

Isi paket:

| File | Fungsi |
|---|---|
| `ergo_rule_config.json` | Semua parameter aturan (satu sumber kebenaran untuk Python dan JavaScript) |
| `ergo_rule.py` | Implementasi referensi Python |
| `ergo_rule.js` | Implementasi JavaScript, logikanya sama persis (browser dan Node) |
| `test_ergo_rule.py` | Uji perilaku dan uji kecocokan Python vs JavaScript (data sintetis) |
| `eval_rule.py` | Menjalankan mesin pada landmark dataset nyata (khusus sisi AI) |

## Pemakaian di browser

```js
const cfg = await (await fetch('ergo_rule_config.json')).json();
const engine = new ErgoRule.RuleEngine(cfg);            // sensitivitas: 'high' | 'medium' | 'low'

// tiap frame dari MediaPipe Face Landmarker:
const ear = ErgoRule.earFromMediapipe(result.faceLandmarks[0] /* atau undefined */, video.videoWidth, video.videoHeight);
const out = engine.update(performance.now() / 1000, ear);   // ear = null bila tidak ada wajah
if (out) render(out);   // satu objek per detik: status, ikm, p50_30, maxclose_30_s, valid_frac_30, alert, events
```

Aturan penting:

- Waktu dalam **detik** dan dari **timestamp frame**, bukan jumlah frame.
- Koordinat landmark dikonversi ke **piksel** (x kali lebar, y kali tinggi) sebelum jarak dihitung. `earFromMediapipe` sudah melakukannya.
- Jangan simpan frame atau landmark. Mesin hanya menyimpan angka EAR beberapa detik terakhir di memori.
- `status`: green, yellow, red, uncertain. Saat uncertain, `reason` berisi calibrating, warming_up, no_face, atau low_valid.
- `events` berisi calibration_done atau calibration_failed. `alert: true` berarti tampilkan peringatan istirahat.
- Kalibrasi 30 detik + warm-up 30 detik: status baru muncul sekitar 60 detik setelah sesi dimulai. Itu sesuai protokol yang dievaluasi.
- Tombol Kalibrasi Ulang: `engine.recalibrate(waktuSekarangDetik)`. Ganti sensitivitas: `engine.setSensitivity('high')`.

## Status parameter

Yang bertanda `provisional` di config (batas merah IKM 66, pembagi skor 0,15, aturan peringatan) adalah tebakan desain, belum ada dasar data.
Jangan ditulis sebagai hasil penelitian. Ambang kuning (IKM 33) punya dasar dari evaluasi awal.

## Menjalankan uji

```
python test_ergo_rule.py       # perlu Python 3 dan, untuk uji kecocokan, Node.js
```
Uji ini memakai data sintetis: membuktikan logika dan kecocokan Python vs JavaScript, bukan akurasi deteksi.

## Cek kecocokan dengan klip nyata (tes paritas browser)

1. Di browser, simpan keluaran (t, ear) dari klip 60 detik ke CSV dengan kolom `t,ear` (ear kosong bila tidak ada wajah).
2. Jalankan `python ergo_rule.py klip.csv` dan `node ergo_rule.js klip.csv`. Keluarannya harus identik.
3. Bandingkan juga EAR dari browser dengan EAR dari Python pada klip yang sama (`extract.py` dengan model `face_landmarker.task` yang sama).
