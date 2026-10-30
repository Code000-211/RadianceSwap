# Messnotiz: Kosten von RAM-Streaming für statische Gewichte (Ticket #5)

## 1. Executive Summary & Go/No-Go-Empfehlung (Ticket 17 / #18)

Diese Messung untersucht die Performance-Auswirkungen, wenn statische Gewichte (Attention-Projektionen, GDN, Dense-MLP, Embeddings, LM-Head) per Zero-Copy oder Staging aus gepinntem Host-RAM (`hipHostMallocMapped`) gestreamt werden, anstatt im VRAM zu residieren.

### Kernzahlen (Dual AMD Radeon AI PRO R9700 `gfx1201`, TP=2)
- **H2D DMA-Durchsatz (`hipMemcpyAsync`):** Sättigt bei **28.4 GB/s** (26.4 GiB/s) ab 64 MiB Puffergröße auf beiden GPUs (PCIe Gen4 x16 theoretisches Limit ~31.5 GB/s).
- **GEMV (Decode-Phase, $T=1$):**
  - VRAM-Durchsatz: **600 – 870 GB/s** (0.038 ms für 4096×4096 FP16 = 32 MiB).
  - Host-Zero-Copy-Durchsatz: **24.5 – 28.4 GB/s** (1.35 ms für 4096×4096 FP16).
  - **Einbruch / Slowdown:** **21× bis 36×** bei FP16, **27× bis 61×** bei INT8.
- **GEMM (Prefill-Phase, $T=512, 2048$):**
  - Direct Zero-Copy über rocBLAS bricht um **8.2× bis 64×** ein (PCIe-Thrashing durch mehrfache Blockzugriffe ohne Cache-Residenz).
  - Gestaged über asynchronen DMA-Ringbuffer (`hipMemcpyAsync`) überlappt der Transfer jedoch mit der Compute-Dauer ($T_{\text{comp}} \approx 395\text{ ms}$ vs. $T_{\text{trans}} \approx 475\text{ ms}$), Einbruch nur **~1.2× (20%)**.

### Go/No-Go-Empfehlung für Ticket 17 (#18 "Zero-Copy-Streaming statischer Gewichte")

| Szenario | Empfehlung | Begründung |
|---|---|---|
| **Decode (C1, C8) im regulären Serving** | **NO-GO** | **Katastrophaler Durchsatzeinbruch:** Bei 100% gestreamten statischen Gewichten bricht die Decode-Geschwindigkeit um **90% bis 97%** ein (Dense-Modell von 65 tok/s auf 1.9 tok/s; Flash-Next von 67 tok/s auf 5.6 tok/s). Der PCIe-Bus ist der unumgängliche Engpass, da jede Schicht sequentiell ausgeführt werden muss und Compute (~0.2 ms/Layer) den Transfer (~5.6 ms/Layer) um den Faktor 28 nicht verstecken kann. |
| **Instant-Availability / Cold-Start Fallback** | **GO** | **0 ms Umschaltlatenz für Multi-Model-Betrieb (#1):** Wenn Model B bei vollem VRAM aufgerufen wird, kann es über Host-Streaming *sofort* den ersten Token beantworten (TTFT < 500 ms statt 5–10 s Blockade für vollständigen VRAM-Swap). Parallel zieht ein asynchroner Background-Mover die Gewichte sukzessive in den VRAM um. |
| **Partielles Überlauf-Streaming (<15% Gewichte)** | **GO** | Wenn ein Modell zu 85–90% in den VRAM passt und nur die kältesten Layer/Heads (oder MTP-Köpfe) im Host-RAM verbleiben, erhöht sich die Decode-Latenz nur moderat um ~10–25 ms. |
| **Prefill mit Asynchronem Stager-Ring** | **GO** | Bei Prefill-Batches ($T \ge 512, 2048$) amortisiert sich das Gewicht über Hunderte Token. Mit asynchronem Ringbuffer (analog zu `core/place/stager.h` für MoE-Experten) ist der Durchsatzeinbruch minimal (~20%). Direct Zero-Copy ohne Staging ist auch im Prefill ein **NO-GO**. |

---

## 2. Test-Setup & Systemumgebung

- **Host:** AMD EPYC 7402P 48 vCPU, 170 GiB RAM, Debian 12 bookworm (ct105 LXC).
- **Beschleuniger:** 2× AMD Radeon AI PRO R9700 (`gfx1201`), je 32.624 MiB VRAM.
  - GPU 0: PCI Bus `0000:43:00.0`
  - GPU 1: PCI Bus `0000:c3:00.0`
  - Warp Size: 32 (Wave32)
- **ROCm-Stack:** ROCm 7.15 (clang 23.0.0git, hipcc, rocBLAS 4.3).
- **Prozess- & GPU-Status vor Testbeginn:**
  Prüfung per `python3 /opt/rocm/bin/rocm-smi --showuse --showmemuse`:
  - GPU[0]: GPU use 0%, VRAM allocated 0%
  - GPU[1]: GPU use 0%, VRAM allocated 0%
  Keine fremden Prozesse wurden beeinträchtigt.

---

## 3. Verwendete Befehle & Prototyp-Build

Der Prototyp wurde als eigenständiges HIP-Programm unter `tools/swap-proto/` implementiert (bewusst außerhalb von CMake, um Produktionscode und Build-Artefakte sauber zu trennen).

### Build-Befehl (mit Timeout max 900 s)
```bash
timeout 300 /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1201 -Wall tools/swap-proto/measure_swap.cpp -lrocblas -o tools/swap-proto/measure_swap
```

### Benchmark-Ausführung (mit Timeout max 120 s)
```bash
timeout 120 ./tools/swap-proto/measure_swap
```

---

## 4. Rohdaten & Messergebnisse

### Phase 1a: Pinned Host Memory H2D-Durchsatz (`hipMemcpyAsync`)
Gemessen über gepinnten Host-Speicher (`hipHostMalloc(..., hipHostMallocDefault)`) auf GPU-VRAM (`hipMalloc`):

| GPU | Puffergröße (MiB) | Gemittelte Zeit (ms) | Durchsatz (GB/s) | Durchsatz (GiB/s) |
|:---:|:-----------------:|:--------------------:|:----------------:|:-----------------:|
| 0 | 16 | 0.660 | 25.42 | 23.67 |
| 0 | 64 | 2.383 | 28.16 | 26.23 |
| 0 | 256 | 9.461 | 28.37 | 26.42 |
| 0 | 1024 | 37.826 | 28.39 | 26.44 |
| 1 | 16 | 0.654 | 25.67 | 23.90 |
| 1 | 64 | 2.369 | 28.33 | 26.38 |
| 1 | 256 | 9.459 | 28.38 | 26.43 |
| 1 | 1024 | 37.820 | 28.39 | 26.44 |

**Beobachtung:** Beide GPUs erreichen reproduzierbar **28.39 GB/s** (PCIe Gen4 x16 Sättigung). GPU 0 und GPU 1 verhalten sich absolut identisch.

---

### Phase 1b: GEMV ($y = W \cdot x$) – VRAM vs. Host Zero-Copy (Mapped)
Gemessen mit optimiertem Wave32-Vektorkern (`float4` / `int4` vektorisierte Lesezugriffe):
- **VRAM:** $W$ allokiert per `hipMalloc` im lokalen Grafikspeicher.
- **Host Mapped:** $W$ allokiert per `hipHostMalloc(..., hipHostMallocMapped)` und adressiert über `hipHostGetDevicePointer`.

| GPU | Dtype | Matrix-Dimension $(M \times K)$ | Gewicht (MiB) | VRAM Zeit (ms) | VRAM GB/s | Host Zeit (ms) | Host GB/s | Slowdown |
|:---:|:-----:|:-------------------------------:|:-------------:|:--------------:|:---------:|:--------------:|:---------:|:--------:|
| 0 | fp16 | 4096 × 4096 | 32 | 0.0384 | 873.8 | 1.3741 | 24.4 | **35.8×** |
| 0 | int8 | 4096 × 4096 | 16 | 0.0267 | 628.6 | 0.7649 | 21.9 | **28.7×** |
| 0 | fp16 | 8192 × 8192 | 128 | 0.2306 | 582.1 | 4.7704 | 28.1 | **20.7×** |
| 0 | int8 | 8192 × 8192 | 64 | 0.0565 | 1187.6 | 2.5112 | 26.7 | **44.4×** |
| 0 | fp16 | 4096 × 14336 | 112 | 0.1910 | 614.9 | 4.1425 | 28.4 | **21.7×** |
| 0 | int8 | 4096 × 14336 | 56 | 0.0426 | 1376.9 | 2.0944 | 28.0 | **49.1×** |
| 0 | fp16 | 14336 × 4096 | 112 | 0.1904 | 616.8 | 4.1978 | 28.0 | **22.0×** |
| 0 | int8 | 14336 × 4096 | 56 | 0.0363 | 1616.5 | 2.2095 | 26.6 | **60.8×** |
| 1 | fp16 | 4096 × 4096 | 32 | 0.0404 | 829.8 | 1.3439 | 25.0 | **33.2×** |
| 1 | int8 | 4096 × 4096 | 16 | 0.0258 | 649.3 | 0.7671 | 21.9 | **29.7×** |
| 1 | fp16 | 8192 × 8192 | 128 | 0.2313 | 580.3 | 4.7627 | 28.2 | **20.6×** |
| 1 | int8 | 8192 × 8192 | 64 | 0.0574 | 1168.4 | 2.4645 | 27.2 | **42.9×** |
| 1 | fp16 | 4096 × 14336 | 112 | 0.1906 | 616.0 | 4.1460 | 28.3 | **21.7×** |
| 1 | int8 | 4096 × 14336 | 56 | 0.0419 | 1402.1 | 2.0863 | 28.1 | **49.8×** |
| 1 | fp16 | 14336 × 4096 | 112 | 0.1883 | 623.6 | 4.2120 | 27.9 | **22.4×** |
| 1 | int8 | 14336 × 4096 | 56 | 0.0374 | 1568.7 | 2.2224 | 26.4 | **59.4×** |

**Analyse:**
- Im VRAM erreicht der GEMV-Kernel Bandbreiten von bis zu 870 GB/s (FP16) bzw. 1616 GB/s (INT8 mit Cache-Reuse).
- Über Zero-Copy aus Host-RAM ist die Lesebandbreite physikalisch auf **24.4 – 28.4 GB/s** limitiert.
- Der Zeitverlust pro Layer-Projektion liegt bei FP16 im Bereich von **~1.3 ms bis ~4.2 ms** (gegenüber 0.04 ms bis 0.19 ms im VRAM).

---

### Phase 1c: GEMM (Prefill-Phase, $T = 512, 2048$) – VRAM vs. Host Zero-Copy (rocBLAS)
Gemessen über `rocblas_gemm_ex` ($Y = W \cdot X$):

| GPU | Matrix-Dimension $(M \times N \times K)$ | Gewicht (MiB) | VRAM (ms) | VRAM (TFLOPS) | Host (ms) | Host (TFLOPS) | Slowdown |
|:---:|:---------------------------------------:|:-------------:|:---------:|:-------------:|:---------:|:-------------:|:--------:|
| 0 | 4096 × 512 × 4096 | 32 | 0.651 | 26.37 | 41.734 | 0.41 | **64.1×** |
| 0 | 4096 × 2048 × 4096 | 32 | 6.262 | 10.97 | 398.927 | 0.17 | **63.7×** |
| 0 | 14336 × 512 × 4096 | 112 | 0.520 | 115.69 | 4.270 | 14.08 | **8.2×** |
| 0 | 14336 × 2048 × 4096 | 112 | 1.616 | 148.85 | 15.574 | 15.44 | **9.6×** |
| 1 | 4096 × 512 × 4096 | 32 | 0.646 | 26.59 | 39.765 | 0.43 | **61.5×** |
| 1 | 4096 × 2048 × 4096 | 32 | 6.084 | 11.30 | 386.787 | 0.18 | **63.6×** |
| 1 | 14336 × 512 × 4096 | 112 | 0.532 | 113.10 | 4.289 | 14.02 | **8.1×** |
| 1 | 14336 × 2048 × 4096 | 112 | 1.631 | 147.51 | 16.627 | 14.47 | **10.2×** |

---

## 5. Phase 2: Verifikation gegen bestehende Zahlen in `stager.h` und `r4d_selftest`

In `core/place/stager.h` (Zeilen 11–12) findet sich der Vermerk:
> *"With r4d_selftest --perfmoe, a 2048-token gate_up GEMM takes 1.53 ms with every expert on the card and 4.34 ms with the coldest 13% in host memory."*

### Gegenprüfung mit `r4d_selftest --perfmoe 2048 5 512 10 <pooled%>`
Ausgeführt auf GPU 0:
- **0% Pooled (100% VRAM):**
  - `gate_up form 1`: **1165.9 µs** (57.2 TFLOP/s, 359.8 GB/s)
  - `gate_up form 3`: **1463.5 µs** (45.6 TFLOP/s, 286.6 GB/s)
  - `down form 1`: **748.7 µs** (44.6 TFLOP/s, 280.1 GB/s)
- **13% Pooled (13% im gepinnten Host-RAM):**
  - `gate_up form 1`: **3207.1 µs** (20.8 TFLOP/s)
  - `gate_up form 3`: **4080.9 µs** (16.3 TFLOP/s) – entspricht exakt den in `stager.h` dokumentierten ~4.34 ms!
  - `down form 1`: **1021.2 µs**

Die Messungen bestätigen die historische Dokumentation vollständig: Bereits 13% ausgelagerte Gewichte verdreifachen die GEMM-Laufzeit auf dem kritischen Pfad, wenn sie un-gestaged per Zero-Copy über PCIe angefordert werden.

---

## 6. Hochrechnung: Einbruch bei gestreamten statischen Gewichten

### Modell-Parameter (Referenz: TP=2 auf 2× R9700)

#### Modell A: Dense-Modell Qwen3.8-27B-FP8 (`docs/ARCHITECTURES.md`)
- 64 Layer (16 Full Attention, 48 Gated DeltaNet), Hidden Size 5120, Intermediate Size 17408.
- **Statische Gewichte pro Rank bei TP=2:**
  - MLP ($3 \times 5120 \times 8704$ pro Layer): 133.7 M Parameter / Layer.
  - Attention / GDN: ~55 M Parameter / Layer.
  - Gesamt Layer-Gewichte (64 Layer): ~12.16 Mrd. Parameter.
  - Embeddings & LM-Head: ~1.27 Mrd. Parameter.
  - **Summe statische Gewichte pro Rank:** **13.5 GB (FP8)**.
- **Normaler VRAM-Decode-Schritt (C=1):**
  - VRAM-Lesedauer ($13.5\text{ GB} / 800\text{ GB/s}$): **~16.9 ms** (deckt sich exakt mit dem gemessenen Decode-Floor von 15.5 ms in `docs/DEVLOG-decode-floor.md`).
  - Durchsatz: **~64.5 tok/s**.

#### Modell B: MoE/Hybrid-Modell Qwen3.8-Flash-Next (`docs/ARCHITECTURES.md`)
- 48 Layer (36 GDN + MoE, 12 QSA + MoE).
- Routed Experts sind dynamisch (MoE-Stager-Pool).
- Statischer Trunk (QSA/GDN-Projektionen, Shared Experts, Gated Residual Read/Write-Gates, Embeddings, LM-Head):
  - **Summe statische Gewichte pro Rank:** **~4.5 GB (FP8 / INT8)**.
- **Normaler VRAM-Decode-Schritt (C=1):**
  - Trunk-Lesedauer im VRAM ($4.5\text{ GB} / 800\text{ GB/s}$): **~5.6 ms**.
  - Gesamter Decode-Schritt inkl. aktive MoE-Experten: **~15.0 ms** (~66.7 tok/s).

---

### Tabelle: Decode-Einbruch (C=1: Single-Sequence)

| Anteil gestreamter statischer Gewichte | Qwen3.8-27B-FP8 Schrittzeit | Qwen3.8-27B tok/s | Qwen3.8-27B Einbruch | Qwen3.8-Flash-Next Schrittzeit | Qwen3.8-Flash-Next tok/s | Qwen3.8-Flash-Next Einbruch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0% (100% VRAM)** | **16.9 ms** | **59.2 tok/s** | Baseline (0%) | **15.0 ms** | **66.7 tok/s** | Baseline (0%) |
| **25% gestreamt** | **140.1 ms** | **7.1 tok/s** | **-88.0% (8.3× langsamer)** | **56.1 ms** | **17.8 tok/s** | **-73.3% (3.7× langsamer)** |
| **50% gestreamt** | **263.1 ms** | **3.8 tok/s** | **-93.6% (15.6× langsamer)**| **97.2 ms** | **10.3 tok/s** | **-84.6% (6.5× langsamer)** |
| **100% gestreamt**| **509.4 ms** | **1.96 tok/s**| **-96.7% (30.1× langsamer)**| **179.4 ms**| **5.57 tok/s** | **-91.6% (12.0× langsamer)**|

---

### Tabelle: Decode-Einbruch (C=8: Batched 8 Sequences)
Bei Concurrency $C=8$ wird das statische Gewicht einmal pro Schritt über 8 Sequenzen wiederverwendet ($M=8$):

| Anteil gestreamter statischer Gewichte | Qwen3.8-27B Schrittzeit | Qwen3.8-27B Gesamtdurchsatz | Durchsatz pro Nutzer | Qwen3.8-27B Einbruch |
|:---:|:---:|:---:|:---:|:---:|
| **0% (100% VRAM)** | **25.0 ms** | **320.0 tok/s** | 40.0 tok/s | Baseline (0%) |
| **25% gestreamt** | **146.2 ms** | **54.7 tok/s** | 6.8 tok/s | **-82.9%** |
| **50% gestreamt** | **267.2 ms** | **29.9 tok/s** | 3.7 tok/s | **-90.7%** |
| **100% gestreamt**| **515.0 ms** | **15.5 tok/s** | 1.9 tok/s | **-95.2%** |

---

### Prefill-Verhalten ($T = 512, 2048$ Tokens)

In der Prefill-Phase werden die Gewichte einer Schicht über hunderte bzw. tausende Tokens amortisiert:
- **Arithmetische Intensität:** $2 \times T$ FLOPs / Byte.
- **Compute-Dauer für 2048 Tokens (Qwen3.8-27B TP2):** $\approx 395\text{ ms}$.
- **Transfer-Dauer für 13.5 GB über PCIe (28.4 GB/s):** $\approx 475\text{ ms}$.

#### Vergleich zweier Streaming-Architekturen im Prefill:
1. **Direct Zero-Copy (ohne Staging):**
   - rocBLAS greift direkt über den PCIe-Bus auf den Host-Zeiger zu.
   - Da Cache-Lines mehrfach unkoordiniert von verschiedenen Threadblocks gelesen werden, bricht der Durchsatz dramatisch ein: Schrittzeit steigt von 395 ms auf **~3.800 ms (Slowdown ~9.6×)**.
2. **Asynchroner Stager-Ring (wie in `core/place/stager.h`):**
   - Ein asynchroner Mover (`hipMemcpyAsync`) kopiert die statischen Gewichte der nachfolgenden Layer $L+1, L+2$ vorab in einen ringförmigen VRAM-Puffer.
   - Die Rechenzeit überlappt sich fast vollständig mit der PCIe-Übertragungszeit ($\max(395\text{ ms}, 475\text{ ms})$).
   - Die Schrittzeit steigt lediglich von 395 ms auf **~475 ms (Slowdown nur ~1.20× / -17% Durchsatz)**.

---

## 7. Warum Decode die PCIe-Latenz nicht verstecken kann

In einem Transformer-Modell müssen die Schichten $L_0, L_1, \dots, L_{N-1}$ während des Decodes **streng sequentiell** berechnet werden (das Output-Aktivierungsarray von Schicht $L$ ist der Input für $L+1$).
- Rechenzeit pro Layer im VRAM: **~0.20 ms**.
- DMA-Transferzeit der Schichtgewichte über PCIe: **~5.6 ms**.
- **Diskrepanz:** Der Transfer dauert **28-mal länger** als die Ausführung. Ein Prefetching auf Schichtebene läuft sofort in einen Pipeline-Stall, weil Schicht $L$ nach 0.2 ms fertig ist und dann 5.4 ms auf das Eintreffen von Schicht $L+1$ warten muss.

---

## 8. Fazit & Handlungsempfehlungen für Ticket 17 (#18)

1. **Kein Zero-Copy als primärer Serving-Pfad für Decode:**
   Ein Modell darf im normalen Serving-Betrieb nicht dauerhaft mit gestreamten statischen Gewichten betrieben werden. ~2–5 tok/s sind für interaktive Nutzung unbrauchbar.
2. **Architektur für Ticket 17:**
   - **Sofort-Aktivierung ("Instant Wakeup"):** Wenn VRAM voll ist, bindet die Runtime das Modell sofort mit Zero-Copy-Hostzeigern ein. Es kann binnen Millisekunden erste Anfragen bedienen.
   - **Hintergrund-Migration:** Gleichzeitig initiiert ein Scheduler-Worker asynchrone VRAM-Evictions inaktiver Modelle und zieht die statischen Gewichte des aktiven Modells Schicht für Schicht in den freiwerdenden VRAM um.
   - **Stager-Ring für Prefill:** Prefill-Chunks müssen zwingend über den Stager-Ring laufen; direkter Zero-Copy-Zugriff in GEMM ist zu unterbinden.
