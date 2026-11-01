# GPU-Abnahme des Model-Swaps (#21)

Gemessen auf 2× AMD Radeon AI PRO R9700 (gfx1201, je 31,86 GiB, 32 357 MiB frei im Leerlauf), TP=2,
Host 48 vCPU / 170 GiB RAM, Build dieses Branches (`-DCMAKE_HIP_ARCHITECTURES=gfx1201`). Die
Messskripte und Manifeste liegen in `scripts/swap-acceptance/` (`start.sh` startet einen Server
im Hintergrund mit `timeout` auf Port 18080 und weigert sich, wenn schon ein `radiance` läuft;
`stop.sh PID` beendet ihn und zeigt den VRAM).

Models: dicht = Qwen3.5-0.8B (HF-Checkpoint, 2,4 GB Gewichte); MoE = Qwen3.8-Flash-Next
(`qwen4exp`, 113,6 GiB `.rad`, davon 67,5 GiB Gewichte + 47,7 GiB gemappte n-gram-Tabelle).
Das MiniCPM-`.rad` (`minicpmv47`) hat in diesem Build kein Architektur-Plugin und startet nicht;
es gibt hier also nur ein MoE-Model, „zwei MoE" sind zwei Einträge derselben Datei.

VRAM-Basis ohne Prozess: 57 MiB je Karte. Ein laufender Prozess ohne aktives Model hält dauerhaft
209 MiB je Karte (HIP-Kontext, Code-Objekte; `kfd` meldet 208 MiB) – das ist der „Rest" in den
Freigabe-Zahlen unten.

## Ergebnis je Punkt

| # | Punkt | Ergebnis |
|---|---|---|
| 1a | Dense + Dense (Routing, kalt, Idle, warm, Tokens) | bestanden |
| 1b | Dense + MoE (dito) | bestanden, nach Fix 3 (vorher war das MoE nie „vorbereitet") |
| 2 | 120 Zyklen Aktivierung/Deaktivierung, TP=2 | bestanden, nach Fix 1 (vorher starb der Server bei der 23. Aktivierung) |
| 3 | Rollback einer gescheiterten Aktivierung | bestanden |
| 4 | Deaktivierung unter VRAM-Druck (`max_active_models` < n) | bestanden (mit Beobachtung: Freigabe-Zahl falsch) |
| 5 | GroupPlanner / Slot-Abgabe mit zwei MoE | Join-Pfad bestanden; Targets änderten sich nicht (laut Code ohnehin `target == offer`, siehe 5); Slot-Shrink/Regrow lief nicht |
| 6 | Heat-Warmup `early_hit` | bestanden: 83,45 % ohne, 89,12 % mit Budget |
| 7 | Parität Manifest vs. `--model` | bestanden (dicht und MoE, Greedy, Sampling mit Seed, Stream, langer Prompt) |
| 8 | Durchsatz / TTFT | gemessen; zwei gleichzeitig aktive Models bremsten sich stark, nach Fix 2 auf rund die Hälfte statt ein Zehntel; TTFT-Aufteilung nur abgeleitet; seit §17 je Request gemessen |
| – | #13 Host-Budget mit großem Model | bestanden nach Fix 3; Host-RAM ist dabei der Engpass (siehe unten) |
| – | Issue-Kriterien Sampling, MTP, Streaming, Abbruch, lange Prompts | bestanden |
| 11 | Gestreamte Statik zurück in den VRAM (#19) | bestanden, siehe §11 |
| 12 | KV-Leihe zwischen Models, die KV-Partition wächst (#17) | bestanden, nach vier Fixes (siehe §12) |
| 13 | Host-Pools vor der Allokation geprüft, MoE `a` + streamendes MoE `b` (#46) | bestanden, kein Exit 137 (siehe §13) |

| 14 | Erschöpfte VA-Arena, Single-Engine-Perf gegen den Stand vor #44 (#45) | bestanden, ein Fehler in `Mover::resize` behoben (siehe §14) |
| 15 | MoE im Manifest mit zwei Models gegen `--model`, die „13-%-Lücke“ (C1) | im früheren Aufbau (`--deterministic`, Headroom 5000) nicht mehr zu reproduzieren (−0,3 %); die einzige reproduzierbare Lücke (−9,2 %) ist der halbierte VRAM-Anteil ohne eigenes `--gpu-headroom-mib`, nicht behoben, Folge-Issue empfohlen (siehe §15) |
| 16 | GroupPlanner unter Dauerlast mit wechselnder Gewichtung, Summe `committed_bytes` ≤ Pool (#51) | bestanden, nach vier Fixes: 7736 Samples, 0 Überschreitungen (vorher 3561 von 4793, bis 3,4 GiB über dem Pool); siehe §16 |
| 17 | TTFT getrennt nach Queue, Aktivierung und Prefill, je Request (`timings`, `radiance:model_ttft_*_seconds`) | gemessen für aktiv, kalt, warm und unter Last mit Warteschlange; die drei Teile summieren sich auf 0,1–0,6 ms genau zum TTFT (siehe §17) |

## Gefundene Fehler und Fixes

1. **Der Server starb bei der 23. Aktivierung** (`core/device/aql.cpp`). Jede Aktivierung legt rund
   46 AQL-Streams an (wahrscheinlich meist die temporären Streams von `hipMemset` in
   `r4d_ar_ipc_alloc`, wenn ein Rank noch keinen Stream hat); die Stream-IDs werden nie
   wiederverwendet und die Tabelle hatte 1024 Plätze (22 × 46 = 1012). Danach scheiterte
   `rad_stream_create`, `hipMemset` meldete nur `hipErrorOutOfMemory`, der Allreduce-Init schlug
   fehl, und die GPU-Gruppe wurde `unhealthy` (alle Requests 503 bis zum Neustart). Reproduzierbar:
   immer nach genau 22 erfolgreichen Aktivierungen (mit `--max-num-batched-tokens 256`, weniger
   Streams je Aktivierung, nach 26). Mit TP=1 tritt es nicht auf (kein Allreduce-Init). Fix:
   `kMaxQueues = 1 Mi` (8 MiB ungenutztes BSS), also etwa 22 000 Aktivierungen je Prozess; die
   Grenze steht als `shortcut`-Kommentar am Code. Regressionstest:
   `device_test: a_process_can_make_more_streams_than_once_fitted_the_id_table` (scheitert mit 1024,
   besteht mit 1 Mi). Danach 120 Zyklen ohne Fehler.
2. **Zwei gleichzeitig aktive Models bremsten sich auf ein Zehntel** (`core/runtime/collective_sequencer.cpp`).
   Dicht + dicht, beide unter Last: C1 je Model 31 tok/s statt 326. Die Übergabe der Collectives
   zwischen den Rank-Threads zweier Engines ging über `condition_variable::wait`; Schlafen und
   Aufwecken kostet auf diesem Host offenbar rund 0,5 ms und fällt 48-mal je Decode-Step an (32 ms je
   Step gemessen). Fix: 300 µs mit `pause` drehen, bevor der Thread schläft. Danach 154 tok/s je
   Model (die Hälfte von ~300, die Grenze der seriellen Reihenfolge). Greedy-Tokens beider Models
   unter gleichzeitiger Last bleiben identisch zu den Einzelläufen (3 Wiederholungen). Kein
   Host-Test: die Wirkung ist Latenz.
3. **Ein MoE mit gemappter n-gram-Tabelle wurde nie „vorbereitet"** (`core/engine_bringup.cpp`,
   `reserve_capture`). Die Kapazitätsprüfung des Stores summierte `stored_bytes` aller Gewichte
   beider Ranks, auch der Mapped-Tier-Tabelle (47,7 GiB je Rank), die nie in den Store kopiert
   wird: 162,9 GiB statt 67,5 GiB, passte nicht neben `--host-headroom-mib` (137,9 GiB verfügbar),
   Warnung „not kept for a warm activation", und das Model blieb nach jeder Deaktivierung kalt
   (volle 113 GiB von der Platte). Jetzt zählen Mapped- und File-Tier nicht mit. Danach: Zustand
   `prepared` (72,47 GB im Store), warme Aktivierung mit `source_read_bytes` 0. Kein Host-Test: die
   Summe steckt in `Engine::load_weights`.

## 1a Dense + Dense

Manifest `m_dd.json`: Model `a` (4096/8 Seqs, bf16-KV, `--deterministic`) und `b` (2048/4 Seqs, fp8-KV),
beide dieselbe 0,8B, `idle_timeout_ms` 8000, TP=2.

- Routing: `model` wählt, kein `model` geht an das Default-Model (`a`), unbekannte Id 404. Zwei
  gleichzeitige Requests an `a` und `b` antworten mit der jeweils eigenen Id.
- Kalte Aktivierung (erster Request an `b`): Aktivierung 2,59 s (`activation_seconds`), davon
  Gewichte 0,687 s (2,69 GB von der Quelle); Request-Wanduhr 2,82 s. Der Start von `a` (Default):
  3,11 s, Gewichte 0,692 s.
- Idle-Deaktivierung nach 8 s: VRAM beide aktiv 3266 MiB je Karte → 1891 (nur `b`) → 300 MiB
  (beide deaktiviert, Prozess lebt); `vram_released_mib` 1587 für `a`; nach dem Beenden 57 MiB.
- Warme Re-Aktivierung: Gewichte 0,245–0,272 s, `bytes_from_store` 2,416 GB, `source_read_bytes` 0,
  `disk_read_bytes` 0; Aktivierung insgesamt 1,56 s im ersten Fall, im 120er-Lauf p50 2,125 s. Vom
  Ganzen sind also nur 0,25 s Gewichte; die übrigen ~1,9 s sind Plan, Allreduce-Init, Arena,
  Kernel-Setup (nicht weiter zerlegt).
- Tokens: `a` kalt = `a` warm = `a` unter Last des anderen = `--model` (md5 über Text und Tokens von
  6 Prompts à 64 Greedy-Tokens, darunter ein 481-Token-Prompt); `b` dito (fp8-KV).

## 1b Dense + MoE

Manifest `m_dm.json`: `moe` (Default; `--placement expert_tiered --host-pool-mib 12288
--gpu-headroom-mib 5000 --kv-cache-dtype fp8 --tp-wire wht6 --num-speculative-tokens 3
--max-num-batched-tokens 2048 --deterministic`) und `d` (0,8B, `--gpu-headroom-mib 300`),
`idle_timeout_ms` 30000.

| Schritt | Messung |
|---|---|
| MoE kalt beim Start | Gewichte 59–70 s (113,6 GiB gelesen, davon 65,9 GiB Container-Sweep mit 2,5–3,1 GB/s, n-gram-Kopie 39,5 s); VRAM 27 579 MiB je Karte |
| `d` kalt (MoE aktiv) | erster Request 3,64 s, Aktivierung 2,11 s; VRAM 29 199 MiB |
| Idle-Deaktivierung MoE | 1,39 s; VRAM 27 636 → 293 MiB je Karte, `vram_released_mib` 27 286; Zustand `prepared`, 72,47 GB im Store, Host-RAM 75 GiB belegt |
| warme Re-Aktivierung MoE | Gewichte 45,2 s (davon n-gram-Kopie 39,1 s von der Platte; die 67,5 GiB aus dem Store ≈ 6 s), `source_read_bytes` 0, `bytes_from_store` 72,47 GB; erster Request 60,0 s (kalt: 70,2 s) |
| `d` warm | erster Request 2,44 s, Gewichte 0,238 s |
| Tokens | MoE kalt = MoE warm = `--model` (`ba5f0d37da96`), `d` = dichter Referenzlauf |

Die warme Aktivierung des MoE spart nur 10 s von 70, weil die gemappte n-gram-Tabelle (Page-Cache-
Backing, nicht im Store; `warm-activation.md`) bei jeder Aktivierung neu nach pinned Speicher
kopiert wird und der Store den Page Cache verdrängt, sodass die Kopie von der Platte kommt
(`disk_read_bytes` 51 GB). Mit `--ngram-placement disk` entfällt die Kopie (Heat-Test unten:
warme Re-Aktivierung 6,3–7,6 s).

## 2 Zyklen (120)

`m_cyc.json` (zwei 0,8B, `idle_timeout_ms` 400), `cycles.py 120`: abwechselnd Request an `a`/`b`,
warten bis alle deaktiviert sind, dann VRAM (`rocm-smi`), Host-RSS, Threads, fds, kfd-Queues messen.

| Größe | Verlauf über 120 Zyklen |
|---|---|
| VRAM Karte 0/1 | 266 MiB / 266 MiB in jedem Zyklus (min = max) |
| kfd-VRAM des Prozesses | 208 MiB, konstant |
| kfd-Queues, Threads, fds | 2, 54, 12, konstant |
| Host-RSS | 2857 MiB (nur `a` im Store) → 5362 (beide) → 5798 (Zyklus 10) → 5849 (44) → 5859 (114) → 5866 MiB (119) |
| Aktivierung warm (n = 118) | p50 2,125 s, p95 2,226 s, min 2,063 s |
| Aktivierung kalt (`b`) | 2,611 s |
| Gewichte warm | p50 0,253 s, p95 0,261 s |

Kein Leck in VRAM, Queues, Threads, fds. Host-RSS steigt nur im ersten Dutzend Zyklen (zweiter
Store-Eintrag und Allokator-Warmlauf) und danach um höchstens 17 MiB in 72 Zyklen (≈ 0,2 MiB je
Zyklus, sprunghaft, nicht linear); ein Leck von der Größe eines Gewichts-Puffers (2,4 GB) oder
eines Pools wäre sofort sichtbar. Ein Leck von wenigen KiB je Zyklus schließen 120 Zyklen nicht aus.

## 3 Rollback

`m_rb.json`: `a` (Default), `nomem` (`--gpu-headroom-mib 32300`, kann nie passen) und `badcfg`
(Pfad auf einen Drafter-Ordner: „container format error").

- Logzeile „the failed activation is rolled back" in beiden Fällen. VRAM danach: `nomem` 32 357 MiB
  frei gegen 32 357 vorher (beim ersten Versuch, während `a` 3000 Tokens erzeugte: 30 766 gegen
  30 770 MiB); `badcfg` 32 357 gegen 32 357. VRAM nach allen Fehlschlägen 266 MiB je Karte
  (nur `a`, vorbereitet).
- `a` lief weiter: die 3000-Token-Generierung während des Fehlschlags lief durch (9,1 s), und
  danach sind `a`s Tokens identisch zum Referenzlauf.
- Backoff: zweiter Request an `badcfg` innerhalb des Backoffs: 503, `Retry-After: 4`.
- Beobachtung: der NOMEM-Fehlschlag von `nomem` verdrängte (nach Ablauf von `a`s Generierung) das
  gesunde `a` (`pressure_deactivations` 1), obwohl `nomem` nie passt; der Request endete nach
  10,1 s mit 503 `model_activation_failed`. Das entspricht der Regel („höchstens eine Runde je
  konfiguriertem Model"), ist aber unnötige Arbeit.

## 4 VRAM-Druck

`m_pr.json`: drei 0,8B (`--max-model-len 262144 --max-num-seqs 64`, damit der KV-Pool den Anteil
wirklich belegt), `max_active_models` 2, Anteil je Model 16 235 MiB.

- VRAM: `a` 16 281 MiB → `a`+`b` 32 544 MiB. Request an `c`: Log „model 'c' needs about 16235 MiB
  of VRAM and the group's tightest card has 79 MiB free" → „found no room in VRAM: deactivating 'a',
  idle the longest". `c` aktiv, `a` vorbereitet. Weitere Wechsel (a, b, c reihum): jeweils das am
  längsten idle Model, `pressure_deactivations` 3/2/2 nach 8 Requests. Wanduhr eines Wechsels
  (Deaktivierung + warme Aktivierung mit 16-GiB-KV): 4,9–5,1 s; Deaktivierung 0,72–0,76 s.
- Fehler in der Anzeige: „model 'a' deactivated: -214 MiB of VRAM freed" – `before` wird beim
  Attach von `a` gelesen, danach aktiviert `b` und belegt 16 GiB, `after − before` ist deshalb keine
  Freigabe von `a`. `vram_released_mib` ist nur bei einem einzelnen Model im Gang richtig (der Log
  sagt „the rest of its use is other models"); nicht behoben.

## 5 GroupPlanner / Slot-Abgabe, zwei MoE

`m_mm.json`: zwei Einträge desselben MoE, jeweils `--host-pool-mib 20000 --ngram-placement disk`,
nicht deterministisch (Heat aktiv). Bei 16 GiB Anteil je Karte reichen 14 000 und 17 000 MiB
Host-Pool nicht („out of memory" beim Start des ersten Models; Experts 5,07 bzw. 2,22 GiB zu
knapp), 20 000 MiB ja.

- Join: beide Models aktiv, VRAM 34,1 GB je Karte; „group planner: 'm2' keeps 10.29 GiB of the
  10.29 GiB of experts it was offered". Beide liefern die Referenz-Tokens (`ba5f0d37da96`) – auch
  nach 80 s Last auf beiden gleichzeitig.
- Last: nur `m2` 2 × 40 s (C8, 128 Tokens, wechselnde Prompts): `m1`s Mover-Zähler
  (`promotions` 8858, `demotions` 9222, `flex_capacity`, `movable`) unverändert – `m1` hat nichts
  abgegeben. Nach dem Code (`group_planner.cpp`) ist das die Konstruktion: der Pool der Gruppe ist
  die Summe der Angebote, kein Model bekommt mehr als sein Angebot, also gilt `target == offer`,
  solange die Experts-Ebene eines Models sein Angebot füllt. Der Shrink-/Regrow-Pfad
  (`slot-release.md`) läuft damit in dieser Konfiguration nicht; er braucht den elastischen Slab
  (#35) oder asymmetrische Angebote. Das Target selbst ist über HTTP nicht abfragbar; belegt ist nur
  „kein Shrink bei `m1`". Vermutlich kühlt die Heat eines idle Models nicht ab (sie klingt je Step
  ab).
- Es gibt keinen Hänger. Ressourcenlage: Host-RAM 160 GiB belegt, 10 GiB verfügbar; nur `m1` wurde
  in den Store übernommen (72,5 GB), `m2` hat keine Store-Kopie.
- Durchsatz siehe 8.

## 6 Heat-Warmup

`m_heat0.json` / `m_heat12288.json` (`mk_heat.py`): ein MoE, `--gpu-headroom-mib 12000`
(VRAM 17,08 GiB, Host-Pool 16,69 GiB, n-gram per `--ngram-placement disk`), nicht deterministisch,
`idle_timeout_ms` 15000. Ablauf: kalter Start, acht verschiedene Prompts à 96 Tokens (Heat lernen),
Idle-Deaktivierung, warme Re-Aktivierung durch dieselbe Arbeitslast, `early_hit` aus `/stats`
(`early_routed_res` / `early_routed`, erste 64 Steps; 246 817 geroutete Experts).

| `warmup_budget_mib` | `early_hit` kalter Start | `early_hit` nach Re-Aktivierung |
|---|---|---|
| 0 | 83,65 % | **83,45 %** |
| 12288 | 83,63 % | **89,12 %** |

Mit Budget: „warmup: 10155 saved-heat experts (12.00 GiB of the 12.00 GiB budget) rank first on
rank 0" (10 139 auf Rank 1). Ohne Budget erholt sich die Trefferrate nicht, mit Budget liegt sie
5,7 Punkte höher. Nach der warmen Re-Aktivierung (Gewichte 6,3–7,6 s) lief die Arbeitslast
27,8–30,0 s statt 10 s beim ersten Start: der Store (67 GiB) hat vermutlich den Page Cache der
gemappten n-gram-Tabelle verdrängt; das ist der Preis eines großen Stores bei 170 GiB RAM, nicht
des Warmups (nicht weiter untersucht).

## 7 Parität und Funktionen

Greedy, 6 Prompts à 64 Tokens (Text und Token-Liste), `--deterministic`:

| Vergleich | Ergebnis |
|---|---|
| dicht `a`: Manifest (kalt, warm, parallel zu `b`) vs. `--model` | identisch |
| dicht `b` (fp8-KV): Manifest (kalt, warm, parallel) vs. `--model` | identisch |
| MoE: Manifest (Headroom 5000, kalt, warm, n = 1 und n = 2) vs. `--model` (Headroom 96 und 5000) | identisch (`ba5f0d37da96`) |
| zwei MoE gleichzeitig, auch nach Dauerlast | identisch |
| Sampling (`temperature 0.8`, `top_p 0.9`, `seed 42`): wiederholt / Manifest vs. `--model` (dicht) | identisch |
| Stream (25 Chunks, `[DONE]`) = Nicht-Stream; lange Eingabe (2421 Tokens), dicht | identisch zu `--model` |
| Abbruch (Verbindung nach 3 Chunks weg), dicht | `in_flight` 0, `running` 0, `client_disconnect_total` 1; das Model wird danach normal deaktiviert |
| MTP (`--num-speculative-tokens 3`, MoE im Manifest) | läuft: Annahmerate 0,75, je Position 0,86/0,85/0,90 |

## 8 Durchsatz und TTFT

C1 = ein Request, 256 Tokens (`ignore_eos`); C8 = acht gleichzeitig, aggregiert; Prefill = ein
Request, ~1800 (dicht, Manifest) bzw. ~3600 Tokens, tok/s aus der Server-Antwort. Je Wert drei
Wiederholungen. Die Spannen sind Minimum bis Maximum.

**Dicht (0,8B)**

| Aufbau | C1 tok/s | C8 tok/s | Prefill tok/s |
|---|---|---|---|
| `--model` (Optionen von `a`; Prefill 3,6k Token) | 325–332 | 2038–2043 | 69 000–71 000 |
| Manifest, `a` allein (`b` kalt) | 326–332 | 2026–2029 | 69 900–71 400 |
| `--model` (Optionen von `b`, fp8-KV) | 328–334 | 1141–1149 | 68 400–71 900 |
| Manifest, `b` allein aktiv (nach `a`) | 296–303 | 1057–1138 | 61 900–72 300 |
| Manifest, `a` aktiv, `b` aktiv aber idle | 294–300 | 1865 | – |
| Manifest, `a`+`b` gleichzeitig **vor Fix 2** | 31 + 31 | 316–331 + 264–273 | 28 000–46 000 je Model |
| Manifest, `a`+`b` gleichzeitig **nach Fix 2** | 154 + 154 | 1287–1366 + 716–725 | 36 000–47 000 je Model (einmal 69 900, als das andere nicht prefillte) |

Ein zweites aktives Model kostet das erste auch im Leerlauf rund 10 % (Tapes werden umgangen,
solange mehrere Engines registriert sind; `ctx.cpp`, „shortcut: tapes are bypassed"). Dass `b` das
auch nach der Deaktivierung von `a` zeigt, wurde nicht geklärt.

**MoE (Qwen3.8-Flash-Next, 3,6k-Token-Prompt)**

| Aufbau | C1 tok/s | C8 tok/s | Prefill tok/s (TTFT) |
|---|---|---|---|
| `--model`, Headroom 96 | 119,6–119,8 | 267–409 | 6540–6695 (543–557 ms) |
| `--model`, Headroom 5000 | 72,4–72,5 | 132–164 | 5040–5140 (692–737 ms) |
| Manifest n = 1, Headroom 5000 | 74,8–74,9 | 144–169 | 5118–5132 (710–713 ms) |
| Manifest n = 2 (`d` kalt), Headroom 5000 | 65,2–65,3 | 123–182 | 4690–4870 (741–782 ms) |
| n = 2, `d` zugleich aktiv und beschäftigt: MoE | 59,3–59,4 | 115–140 | 4520–4730 (762–793 ms) |
| n = 2, dasselbe: `d` | 52,1–52,2 | 111–135 | 28 000–64 600 (57–127 ms) |
| `d` allein bei aktivem, idlem MoE | 295–300 | 1863–1868 | 67 400–69 200 |

Der Unterschied n = 1 zu n = 2 (rund 13 % bei C1, in drei getrennten Starts reproduzierbar) ist
unerklärt: `d` war dabei nie aktiv, die Optionen des MoE sind gleich.

**Zwei MoE** (`m_mm.json`, beide aktiv, 20 GiB Host-Pool je Model, Host-RAM voll): C1 9,3 → 11,5 →
30,4 tok/s je Model (die Heat wärmt sich); C8 52–63 tok/s je Model; Prefill 876–1324 (`m1`) und
605–616 (`m2`) tok/s mit TTFT 2,2–4,9 s. Eine C8-Last aus 8 Clients über 40 s: `m1` allein
102 tok/s, beide zugleich 38,4 + 35,3. Ein Teil davon ist der Host-RAM (Pools und Store verdrängen
den Page Cache), ein Teil die Collective-Reihenfolge; getrennt wurde das nicht.

**TTFT nach Queue / Aktivierung / Prefill.** Je Request gemessen steht es in §17. Hier die frühere
Ableitung (damals gab es keine Metrik für Queue und Aktivierung): Request-Wanduhr = Router-Wartezeit + `activation_seconds` (aus
`/radiance/models`) + `timings.ttft_ms` der Engine.

| Fall | Wanduhr | Aktivierung | Prefill-TTFT (Engine) | Rest (Queue, Tokenisierung) |
|---|---|---|---|---|
| dicht, warm (120 Zyklen) | p50 2,158 s, p95 2,255 s | p50 2,125 s, p95 2,226 s | ~25 ms | ~8 ms |
| dicht, kalt | 2,645 s | 2,611 s | ~25 ms | ~10 ms |
| MoE, warm (`ngram ram`) | 60,0 s | ~59 s | 0,76 s (3,6k Token) | – |
| MoE, kalt | 70,2 s | ~69 s | 0,76 s | – |

## Beobachtungen, nicht behoben

- **Host-RAM ist die Grenze großer Models (#13).** Ein MoE aktiv und vorbereitet: 153 GiB belegt,
  17 GiB verfügbar, Page Cache auf 17 GB gedrückt. Zwei MoE: 160 GiB belegt, 10 GiB verfügbar. Der
  Store nahm nur das erste auf. `cache.host_budget_mib` wurde nicht variiert.
- `/stats` und `/metrics` melden nur das Default-Model (`?model=` wird ignoriert); die Heat/Mover-
  Zähler eines zweiten Models sind nur im Log sichtbar.
- Ein Allreduce-Init, der an `hipMemset` scheitert, meldet nur „hip: out of memory" und verbirgt
  „more than N streams in one process" (`aql.cpp`, `hipMemset` bildet jeden Fehler auf
  `hipErrorOutOfMemory` ab).
- `ctest -L fast`: acht Tests schlagen in dieser Umgebung fehl (`arch_test`, `gbnf_test_shard0/2`,
  `images_test`, `kbench_mutant_test`, `kbench_test`, `tokenizer_test`, `videos_test`); mit
  `origin/main` nicht verglichen, keiner nennt eine geänderte Datei (Architektur-Vokabular,
  Grammatik, Tokenizer-/Bilddaten). Die Swap-Tests (`model_*`, `weight_store`, `manifest`,
  `collective_sequencer`, `group_planner`) und `device_test` bestehen.
- Nicht geprüft: MoE/MoE mit `cache.host_budget_mib`, mehr als zwei Models, Streaming unter
  gleichzeitiger Deaktivierung eines zweiten Models, 120 Zyklen für das MoE (eine Aktivierung
  dauert 60–70 s), Sampling/Stream/Abbruch/langer Prompt am MoE.

## 9 Elastischer Expert-Slab (#35)

Zwei Einträge desselben MoE (`m_el.json`: wie `m_mm.json`, aber `--host-pool-mib 26000`, damit der Floor
unter dem Angebot liegt), TP=2, `idle_timeout_ms` 0. Skripte: `elastic.py` (Last nur auf ein Model, VRAM
und `/radiance/models` je Phase, Greedy-Tokens beider Models danach). Das Angebot (Pool = Summe) ist je Rank
20 680 / 20 773 MiB; die Spalten sind `committed / target / offer` des Expert-Slabs in MiB.

| Phase | Rank | m1 | m2 | Σ committed ≤ Pool |
|---|---|---|---|---|
| Ende `m1:50` (nur m1 unter Last) | 0 | 13 448 / 13 449 / 10 314 | 7 231 / 5 168 / 10 366 | 20 679 ≤ 20 680 |
| | 1 | 14 327 / 14 328 / 10 361 | 6 445 / 5 192 / 10 412 | 20 772 ≤ 20 773 |
| Beginn `m2:60` (Übergang) | 0 | 8 613 / 7 042 / 10 314 | 8 894 / 12 067 / 10 366 | 17 507 ≤ 20 680 |
| | 1 | 13 513 / 7 037 / 10 361 | 6 572 / 7 260 / 10 412 | 20 085 ≤ 20 773 |
| Ende `m2:60` (nur m2 unter Last) | 0 | 7 220 / 5 168 / 10 314 | 13 460 / 13 460 / 10 366 | 20 680 ≤ 20 680 |
| | 1 | 6 445 / 5 192 / 10 361 | 14 319 / 14 328 / 10 412 | 20 764 ≤ 20 773 |

- Unter m1-Last hält m2 rund 3 GiB je Rank weniger als sein Angebot (committed 7 231 gegen 10 366 auf Rank 0), und m1
  committet sie (13 448 gegen sein Angebot von 10 314); unter m2-Last kehrt es sich um (m2 13 460, m1 7 220).
  Die Summe der Commits übersteigt den Pool nie (die Planer-Gate vergibt Wachstum erst nach der Meldung
  des Gebers). Das Target des Gebers fällt sofort (5 168), das des Nehmenden folgt der Freigabe
  (Zeile "Beginn `m2:60`": m2-Target 12 067 gegen committed 8 894, Rest folgt in 128-MiB-Schritten).
- `rocm-smi` je Karte: 32,4–32,5 GiB belegt am Phasenende, zwischendurch bis 26,5 GiB (die Freigabe ist
  sichtbar, die Summe pro Karte allein belegt aber nicht, wer sie bekommt: dafür `committed` je Engine).
- Greedy-Tokens beider Models nach jeder Phase identisch zur Referenz (`5e8678d0682f`); keine Hänger, 0
  Fehler in 897 / 1282 Tokens unter Last.
- Rampe: ein Schritt Wachstum (128 MiB) kostet das Entmappen/Mappen von 64 Granulen; 5 GiB Umschichtung
  dauern wenige Sekunden. Ein Phasenwechsel erreicht das Ziel innerhalb der ersten 20–30 s (Idle-Heat der
  Gegenseite kühlt in 20 s auf kalt).

**Bekanntes Risiko, nicht behoben.** Bei mehreren Läufen (Varianten mit echtem Entmappen/Wiedermappen,
während die andere Engine Seiten zurückgibt) lieferte eine Greedy-Anfrage mit `logprobs` kein endliches
Logit (`row 0 has no finite logit`, HTTP 500); der vollständige Lauf oben (alle Phasen, alle Digests)
war fehlerfrei. Ausgeschlossen durch Gegenproben: ein Leser eines freien Slots (Vergiften der freigegebenen
Slots statt Entmappen: kein Fehler), nur logisches Schrumpfen (kein Fehler), Stream-Synchronisation vor dem
Entmappen, Entmappen nur im Leerlauf, Kopien über Blit-Kernel statt der DMA-Liste (der Fehler blieb).
Es bleibt der Verdacht auf das Entmappen/Wiedermappen derselben Adresse unter Last (Treiber/TLB). Bis das
geklärt ist, ist die Arbeit hinter der Manifest-Bedingung (mehr als ein Model) zu bewerten, nicht als
Dauerbetrieb zu empfehlen; die Single-Engine-Pfade sind nicht betroffen.

### Single-Engine (ohne Manifest), vorher / nachher

`--model`, TP=2; `origin/main` gegen diesen Branch, je drei Wiederholungen. Der Pool ist dort unverändert
eine `rad_dev_alloc`; der Mover reserviert wie zuvor.

| Aufbau | C1 tok/s | C8 tok/s | Prefill tok/s |
|---|---|---|---|
| dicht 0,8B, main | 325–332 | 2032–2056 | 57 000–65 100 |
| dicht 0,8B, Branch | 326–333 | 2015–2039 | 57 300–64 700 |
| MoE `--deterministic`, main | 73,8–77,7 | 129–151 | 4645–4866 |
| MoE `--deterministic`, Branch | 72,8–89,0 | 137–226 | 4697–5150 |
| MoE dynamisch, main | 119,6–181,5 | 298–702 | 3183–3377 |
| MoE dynamisch, Branch | 119,2–183,0 | 267–552 | 3078–3331 |

C1 und C8 liegen im dichten Fall innerhalb 1 %. Prefill streut bei beiden Builds von Lauf zu Lauf um
±5 % (Basis allein 57 257–65 110); ein systematischer Unterschied ist nicht messbar. Die MoE-Läufe
streuen durch die Heat (dynamisch) und die Kartenlage stärker als 2 %; ein Unterschied zwischen den Builds ist darin nicht zu erkennen (nicht statistisch geprüft).

## 10 Streaming statischer Gewichte (#18)

`scripts/swap-acceptance/static_streaming.py MODEL --seqs 8 --pct 25|50|100` (Port 18080, GPUs
exklusiv, `--max-model-len 4096`). Phasen: `ref` (alles auf der Karte, bzw. beim MoE der Normalfall mit
Pool), `stream` (Headroom so, dass `pct` % der Statik streamen; Greedy-Text und -Tokens wie `ref`),
`status` (Ein-Model-Manifest: Anteil in `/radiance/models` und `/metrics`), `full` (Model `a` hält die
Karten, `b` aktiviert sich bei vollem VRAM). Dicht = Qwen3.5-0.8B (`a` dasselbe Model, 262144/64);
MoE = Flash-Next (`--host-pool-mib 12288 --ngram-placement disk`, `a` = die 0,8B, weil zwei MoE nicht in
den Host-RAM passen). Die Statik je Rank: 910 MiB (dicht), 3666 MiB (MoE).

| Prüfpunkt | Dicht (25 / 50 / 100 %) | MoE (25 / 50 / 100 %) |
|---|---|---|
| Anteil gestreamt (`stream`; `status` = Plan) | 25,6 / 50,8 / 85,5 % (100 %: Obergrenze mit 8 Sequenzen, ein Cache und der Ring bleiben; mit 1 Sequenz 99,998 %) | 15,4 / 42,7 / 96,5 % |
| Ausgabe = Referenz (stream, Manifest, full) | ja, 9 von 9 | ja, 9 von 9 |
| `/radiance/models`, `/metrics` zeigen den Anteil | ja (`static_streamed_fraction` 0,256 / 0,508 / 0,855) | ja (0,154 / 0,427 / 0,965) |
| Aktivierung bei vollem VRAM | gelingt, `b` streamt (6 / 33 / 84 %), `a` läuft weiter und antwortet gleich | gelingt, `b` streamt (15,8 / 43,0 / 96,7 %), `a` ebenso |
| Aktivierung / erste Antwort von `b` | 2,1–2,3 s / 2,2–2,6 s | 35,5–37,0 s / 36,2–37,7 s |

Kosten (drei Wiederholungen, Spanne; C8 = acht gleichzeitig; Prefill 3,6k Token):

| Model | Anteil | C1 tok/s | C8 tok/s | Prefill tok/s (TTFT ms) |
|---|---|---|---|---|
| dicht | 0 % | 323–330 | 1970–2001 | 67 500–68 600 (53) |
| | 25,6 % | 198–199 | 1337–1345 | 60 800–61 700 (59–61) |
| | 50,8 % | 62,6–62,9 | 474–476 | 42 200–43 200 (84–86) |
| | 85,5 % | 32,4–32,5 | 226–247 | 31 500–34 500 (105–115) |
| dicht, 1 Sequenz | 52,4 % / 26,3 % / 99,998 % | 55,6 / 167 / 25,6 | = C1 (eine Sequenz) | 40 500 / 57 000 / 26 000–26 700 |
| MoE | 0 % (Experts teils im Pool; Heat wärmt sich: 139 → 181) | 139–181 | 569–656 | 5715–6279 (579–631) |
| | 15,4 % | 36,5–37,4 | 61–74 | 2713–2846 (1273–1327) |
| | 42,7 % | 23,5–25,0 | 48–70 | 2245–2485 (1447–1590) |
| | 96,5 % | 11,2–11,5 | 42–65 | 2067–2212 (1665–1755) |

Beim MoE ist der Einbruch Statik plus Experts: sobald die Statik streamt, bekommt die Karte keinen
Platz mehr für Experts (sie liegen alle im Pool), der Vergleich zur Referenz mischt beides.

### Gefundene Fehler und Fixes

1. **`--deterministic` ließ gestreamte Statik zu** (ADR 0002). Ohne Stager liest der Prefill die
   Statik über PCIe an Ort und Stelle. Jetzt Startfehler: `--deterministic: 438.05 MiB of the static
   weights do not fit the card beside one sequence of cache … has no prefill stager to copy them
   ahead` (Exit 1, VRAM danach auf der Basis; Test `deterministic_refuses_streaming_statics`).
2. **`static_streaming.py` Phase `full` bei 100 %** zielte auf 34 MiB Rest für `b`; die festen Kosten
   von `b` (Arena, Staging, Queues) fehlten um 19 MiB, `b` scheiterte mit „VRAM does not cover the fixed
   costs“ und der Druckpfad deaktivierte `a` (korrekt, aber nicht das, was die Phase prüft). Das Ziel hat
   jetzt 64 MiB Reserve. Neu: `--seqs` (C8 braucht 8 Sequenzen, mit 1 war C8 = C1), `--extra`, `--hog`, `--hog-extra`
   (ein kleines `a` neben dem MoE); `start.sh` wartet bis zu 60 s, bis ein eben beendeter Server
   (großer MoE gibt seinen gepinnten Speicher langsam frei) weg ist, statt „REFUSE“ zu melden.

Rebase auf `origin/main` (#44, elastischer Slab): keine Textkonflikte; Build, `ctest -L unit` (nur die
8 bekannten rot) und alle Läufe oben auf dem rebasierten Stand. Im Manifest (`status`, `full`) läuft
der GroupPlanner des Slabs mit: ein streamendes `b` tritt mit „offering 0 B of experts“ bei (die Karte
hat nach der Statik keinen Platz für Experts), `a` bleibt unverändert. Nicht gemessen: zwei MoE, bei
denen `a` Experts im Slab hält und `b` streamt. Der Versuch (`a` = MoE mit 20 GB Pool, `b` = MoE, 50 %)
scheiterte am Host-RAM: `a` allein belegt 138 GiB, die Pool-Erhöhung von `b` (+20 282 MiB je Rank) kam
dazu, der Kernel beendete den Server (Exit 137, ohne Meldung, `host budget` warnte nicht). Dieselbe
Grenze wie „Host-RAM ist die Grenze großer Models (#13)“ oben; behoben in #46, siehe §13.

## 11 Gestreamte Statik zurück in den VRAM (#19)

`scripts/swap-acceptance/static_streaming.py MODEL --seqs 8 --pct 25|50|100 --phases promote --idle-ms 30000`
(Port 18080, GPUs exklusiv; `ref` muss vorher gelaufen sein). Design: `static-streaming.md`, Abschnitt
„Zurück in den VRAM“. Dicht = Qwen3.5-0.8B (910 MiB Statik je Rang); `a` dasselbe Model (262144/64) hält die
Karten und lässt `b` Platz für `pct` % der Statik (wie Phase `full`); `cache.idle_timeout_ms` 30 000.

Ablauf der Phase: `b` aktiviert sich bei vollem VRAM und streamt, Greedy-Ausgabe (`gen.py`) gleich der
Referenz; vier gleichzeitige `logprobs`-Anfragen 15 s lang (Last auf den gestreamten Gewichten); `bench.py`
(C1, C8, Prefill 3,6k Token, je drei Wiederholungen) mit `a` aktiv, das ein winziger Request alle 7,5 s wach hält;
dann bekommt `a` keine Requests mehr und `b` genau eine `logprobs`-Anfrage nach der anderen (Keep-alive und
Genauigkeitsprobe: allein im Batch muss dasselbe Prompt zu jedem Zeitpunkt denselben Text geben); `a` läuft nach dem
Timeout in die Deaktivierung, der Anteil von `b` wird jede Sekunde gelesen; danach wieder vier gleichzeitige
`logprobs`-Anfragen, `gen.py` gegen die Referenz, `bench.py`.

| Prüfpunkt | 83,9 % | 32,7 % | 6,4 % |
|---|---|---|---|
| `b` streamt, während `a` die Karten hält (Ausgabe = Referenz) | ja | ja | ja |
| Anteil nach der Deaktivierung von `a` (`/radiance/models`, `/metrics`) | **0** (`static_streamed_bytes` 0) | **0** | **0** |
| Sekunden von der Deaktivierung von `a` bis Anteil 0 | 2,0 | 1,0 | 1,0 |
| VRAM je Karte, `a` und `b` aktiv → `b` allein, zurückgeholt | 31 831 → 1760 / 1789 MiB | 31 377 → 1720 / 1749 | 31 149 → 1698 / 1727 |
| `gen.py` nach dem Zurückholen = Referenz, Text und Tokens | ja | ja | ja |
| Anfragen während des Übergangs (eine nach der anderen, `logprobs`) | 135, 0 Fehler, 0 nicht endlich, alle Texte gleich | 198, 0, 0, gleich | 309, 0, 0, gleich |
| vier gleichzeitige `logprobs`-Anfragen, streamend / zurückgeholt (je 15 s) | 60 / 516 Anfragen, 0 Fehler, 0 nicht endlich | 188 / 508, 0, 0 | 472 / 508, 0, 0 |
| „no finite logit“ im Server-Log | 0 | 0 | 0 |

Ohne Fehler und ohne nicht endliche Logits: das Zurückholen mappt nur frische Adressen und gibt keine
frei (`Pool::va_arena`); anders als der Expert-Slab (#35) entmappt es nichts, das Aliasing-Problem der
Wiederverwendung entsteht nicht. Mehrere gleichzeitige Anfragen liefern dieselbe Prompt gelegentlich anders
(vier Threads, Batch-Zusammensetzung wechselt): 1–2 abweichende Antworten je 500 Anfragen, auf der Basis ohne
Streaming 2 in 1028 (`--phases ctl`); die Probe auf Bit-Gleichheit ist deshalb die einzelne Anfrage und `gen.py`.

Durchsatz (Median von drei; C8 = acht gleichzeitig; Prefill 3,6k Token; Referenz = ein Model, alles auf der
Karte, Phase `ref`):

| Anteil von `b` | Zustand | C1 tok/s | C8 tok/s | Prefill tok/s (TTFT ms) |
|---|---|---|---|---|
| – | Referenz (ein Model, kein Streaming) | 333,3 | 2053,3 | 70 000 (52) |
| 83,9 % | streamt, `a` aktiv | 32,1 | 252,8 | 33 400 (109) |
| 0 % | zurückgeholt, `a` deaktiviert | 331,7 | 2016,9 | 62 700 (59) |
| 32,7 % | streamt, `a` aktiv | 108,7 | 771,9 | 49 400 (74) |
| 0 % | zurückgeholt | 330,2 | 2013,1 | 66 800 (55) |
| 6,4 % | streamt, `a` aktiv | 301,5 | 1862,4 | 67 000 (55) |
| 0 % | zurückgeholt | 331,3 | 2017,6 | 66 500 (56) |
| 0 % | Kontrolle: `b` streamt nichts, `a` aktiv (Manifest) | 302,5 | 1873,0 | 67 600 (56) |

Der Decode erholt sich auf das Niveau ohne Streaming (99 % der Referenz bei C1, 98 % bei C8): 3,0× bei 33 %,
10,3× bei 84 % (C1). Der Prefill erholt sich auf 89–95 % der Referenz (nach 84 %: 62 700 gegen 70 000, nach 33 %
und 6 %: 66 500–66 800); der Rest der Lücke ist nicht aufgeklärt (die Kontrolle mit `a` aktiv liegt bei 67 600, der
Streaming-Zustand selbst kostet ihn also nicht; die Lücke ist beim höchsten gestreamten Anteil am größten). Mit `a`
aktiv kostet das Manifest allein rund 9 % im Decode (die Kontrolle: 302,5 / 1873 gegen 333 / 2053), das ist die
Mehr-Model-Last aus §8 und kein Effekt des Zurückholens: nach der Deaktivierung von `a` ist sie weg.

Beobachtungen:

- Die Zeit bis Anteil 0 ist die Länge der H2D-Kopie je Rang (0,9 GiB; DMA-Liste mit `RAD_COPY_YIELD`) plus
  ein `group_tick` (100 ms); der Anteil fällt in 1–2 s nach der Deaktivierung. Der Schub je Tick ist 256 MiB.
- Schon beim Start nutzt `b` die wenigen MiB, die über dem Boden frei sind (10 MiB je Rang bei 32,7 %): das Log
  zeigt den ersten `promoted static weights`-Schub vor der Deaktivierung von `a`.
- Nicht gemessen: ein MoE (Host-RAM ~170 GiB; zwei MoE führen zum OOM-Kill, Issue #46) und der Fall „Expert-Slab
  gibt Seiten ab“ auf der Karte (Host-Test: `group_planner_test`, kein Doppelverplanen mit den Expert-Zusagen).

## 12 Modellübergreifende KV-Leihe: die KV-Partition wächst (#17)

`scripts/swap-acceptance/kvgrow.py LOG SEQS PROMPT_TOKENS GEN_TOKENS [IDLE_S]` mit `m_kv.json`, Server
`start.sh LOG 1500 --manifest scripts/swap-acceptance/m_kv.json --tp 2` (Port 18080, GPUs exklusiv). Design:
`kv-lend.md`. Beide Models sind dieselbe Flash-Next-MoE-Datei (zwei MoE, Host-RAM beachtet: `--host-pool-mib`
26 000 und 20 500; kein OOM). **A** = `m1`, `--max-model-len 4096 --max-num-seqs 4`,
`--host-pool-mib 26000`, hält nach 20 s Last den vollen Expert-Slab (S0: 10 051 / 10 100 MiB committed je Rang, sein
Angebot sind 10 394 MiB). **B** = `m2`, `--max-model-len 16384 --max-num-seqs 16`, `--host-pool-mib 20500`: der kleine
Host-Pool macht den Expert-Boden hoch (9,72 GiB = sein ganzes Angebot) und lässt dem Cache nur 1916 MiB Carve
(21 112 Token + 1,7 GiB Zustände für 16 Sequenzen), seine Obergrenze ist 3788 MiB. Ein Stück = 234 MiB = 32 768
Token. (Die ursprüngliche `m_kv.json` ließ B's Carve die ganze Obergrenze decken: 5,38 GiB, dann gab es nichts zu
wachsen.)

Ablauf: Referenz (jedes Prompt allein auf B, Greedy, `logprobs`), A 20 s warm, dann alle Prompts gleichzeitig auf B
(`logprobs`, 96 Token), 30–45 s Leerlauf, je eine Probe an A und B. `rocm-smi --showmeminfo vram` und
`/radiance/models` alle 2 s (der Beobachter, der das Aliasing in #35 auslöste).

| Lauf | Last auf B | größtes Wachstum (je Rang) | Anfragen B / Fehler / nicht endlich | `never_twice` | Rückgabe |
|---|---|---|---|---|---|
| 1 | 12 × ~6000 Token | 468 MiB (2 Stücke) | 12+12+2 / 0 / 0 | ja | auf 0 nach ~13 s, A+B committed 20 011 / 20 094 gegen S0 20 071 / 20 119 MiB |
| 2 | dito | 468 MiB | dito | ja | auf 0, 20 016 / 20 046 gegen 20 071 / 20 119 |
| 3 | dito | 468 MiB | dito | ja | auf 0, 20 011 / 20 046 gegen 20 071 / 20 119 |
| 4 | 16 × ~8000 Token, 64 Token | 702 MiB (3 Stücke) | 16+16+2 / 0 / 0 | ja | auf 0 nach ~20 s, 20 016 / 20 094 gegen 20 071 / 20 082 |

Zeitlauf von Lauf 1 (Sample alle 2 s; A committed | B committed | B Wachstum gewährt/gebackt, Rang 0; MiB):

| t | A | B | B's KV (gebackt / Ziel) | Beobachtung |
|---|---|---|---|---|
| 103 s (S0) | 10 051 | 10 019 | 0 / 0 | A warm, voller Slab |
| 109,8 s | 10 019 | 10 019 | 0 / 151 | B's Anspruch, A's Slab schrumpft noch nicht |
| 116,2 s | 9 348 | 10 276 | 234 / 468 | A gibt ab, das erste Stück gebackt |
| 122,6 s | 7 813 | 11 657 | 468 / 468 | beide Stücke gebackt |
| 129 s | 7 238 | 12 271 | 468 / 0 | Last zu Ende, der Anspruch endet |
| 141,8 s | 7 238 | 12 726 | 0 / 0 | beide Stücke zurück, die 468 MiB bei B's Experts |
| 154,5 s | 7 622 | 12 389 | 0 / 0 | Leerlauf, beide kalt |

- **B bekommt KV, obwohl A den VRAM füllt**: ohne Wachstum kann B's Carve (21 112 Token) die Last nur nacheinander
  bedienen (Kontrolle mit `RADIANCE_ELASTIC_SLAB=off`: kein Stück); mit Wachstum
  wurde in jedem Lauf das erste Stück 6–7 s nach dem Anspruch gebackt, nachdem A's Slab die Bytes zurückgegeben hatte, das
  zweite 6 s später.
- **A's committed Slab sinkt**: 10 051 / 10 100 → 7 238 / 6 613 MiB (−2,8 / −3,5 GiB je Rang). Das ist nicht nur der
  KV-Anspruch: B's Experts sind heiß und wachsen auch (10 019 → 12 271 / 12 989 MiB, #35). Der KV-Anteil sind die
  468–702 MiB der Stücke; die Summe aus A's Slab, B's Slab und B's Stücken blieb in jedem Sample unter dem Pool
  (`never_twice`; pro Rang, 20 267 MiB Angebote).
- **Laufende Sequenzen ungestört**: alle 12 (16) Anfragen antworteten 200 mit endlichen Logprobs. Die Greedy-Tokens unter Last weichen nach
  5–83 Token von der Einzel-Referenz ab (9–10 von 12 Prompts); **dasselbe zeigt die Kontrolle ohne Wachstum**
  (`RADIANCE_ELASTIC_SLAB=off`: 9 von 12 nach 5–83 Token). Vor der ersten Abweichung ist die größte |Δ logprob| gegen die
  Referenz 0,32 / 0,42 / 0,33 / 0,31 (Läufe 1–4) mit Wachstum und 0,33 ohne: Batchrauschen. Verschobene KV-Bytes
  zeigten sich schon beim ersten Token; das erste Token stimmte immer.
- **Keine nicht endlichen Logits** in vier Läufen mit `rocm-smi` alle 2 s als Beobachter (in jedem Lauf `logprobs` auf jeder Anfrage an B und A; Server-Log: „no finite logit“ 0, kein `E`/`W` zum KV). Die Stücke mappen nur frische
  Adressen und geben nie eine zurück (`Pool::va_arena`), das Aliasing aus §9 kann nicht entstehen.
- **Rückgabe nach Idle**: B's Wachstum fällt 5 s nach dem Ende des Anspruchs stückweise auf 0 (ein Stück alle 5 s: 2 Stücke
  binnen ~13 s, 3 Stücke binnen ~20 s); die Bytes gehen an die Experts (A+B committed liegt am Ende wieder bei S0, ±100 MiB).
  VRAM je Karte nach dem Server-Ende: 57 MiB (Basis).

### Fehler und Fixes

1. **Wartende Prompts waren kein Bedarf** (kein Wachstum: `held` blieb unter dem Carve). `want` war `held + Puffer`,
   und ein Prompt nimmt seine Blöcke erst Chunk für Chunk, ein wartender Request hält keine; also sah weder `held` noch der Fehlbetrag-Spitzenwert den Bedarf. Fix: `Scheduler::unplaced_tokens_locked`
   (Sequenzlänge minus berechnete Token aller laufenden und wartenden Requests) geht in den Anspruch ein.
2. **Der Carve verbarg, wie voll die Blöcke waren.** `want` (in Bytes) wurde mit dem ganzen Carve verglichen, der 1,7
   GiB Sequenzzustände enthält, die nie wachsen; `held + Puffer + wartend` blieb auch voll belegt unter dem Carve. Fix:
   `KVManager::tokens_beyond_carve` rechnet je wachsender Gruppe in Blöcken gegen deren Carve-Blöcke, in ganzen
   Stücken.
3. **Der Prefix-Cache hielt das Stück und den Anspruch fest** (B's Wachstum blieb 45 s nach dem Ende bei 468 MiB). Blöcke
   fertiger Gespräche zählen im `held`; sie sind Cache, keine Last. Fix: Blöcke nur des Prefix-Cache zählen nicht zum
   Bedarf (`cache_only_blocks`), und die Rückgabe räumt bis zu 4096 Cache-Einträge pro Versuch
   (`Scheduler::evict_cache_locked`), wenn nur sie das oberste Stück halten.
4. **Zusammenführen mit #19 (Statik in den VRAM)**: beide nehmen die freien Bytes der Karte. `group_statics` rechnete
   einen gewährten, noch nicht gebackten KV-Anspruch nicht als vergeben, die Statik hätte dieselben Bytes bekommen. Fix:
   `kv_grow − grown` zählt wie der Grant eines elastischen Slabs; die Reihenfolge ist Statik, KV-Anspruch,
   Expert-Wachstum (`kv-lend.md`).
5. Skript: `kvgrow.py` schickte A (`--max-model-len 4096`) Prompts von B's Länge (HTTP 400); `returned` verlangte, dass A
   allein auf S0 zurückkehrt (B's heiße Experts behalten den Teil, der KV-Anteil ist aber zurück), jetzt A+B gegen S0;
   `running_undisturbed` verlangte Gleichheit der Tokens, die auch ohne Wachstum nicht gilt, jetzt Logprobs vor der
   ersten Abweichung (siehe oben).

### Grenzen

- Die Kette Anspruch → A's Slab schrumpft → Stück gebackt dauert 6–7 s je Stück; solange bleibt B beim Carve.
- B's Carve hängt von `--host-pool-mib` ab (Expert-Boden); das Manifest wurde darauf eingestellt, dass der Carve kleiner
  ist als die Obergrenze. Eine Obergrenze, die der Carve ohnehin deckt (Standard `m_kv.json` zuvor), wächst nie.
- Nicht gemessen: Idle KV von A an den Slab von B (`lend_from` über Engines, bleibt ungebaut), drei Models, dichte
  Models mit großem Cache-Bedarf.

## 13 Host-Pools vor der Allokation (#46)

`scripts/swap-acceptance/host_budget_stream.py MODEL.rad --a-headroom-mib 2836 --extra "--placement expert_tiered --host-pool-mib 26000 --kv-cache-dtype fp8 --tp-wire wht6 --ngram-placement disk"`
(Port 18080, GPUs exklusiv, Merge-Stand mit `origin/main` inkl. #47 und #48). Beide Models sind das MoE
(`qwen4exp`, 65,8 GiB Gewichte je Container-Kopie); `--host-pool-mib 26000` je Rang, damit `a` einen großen Host-Pool hält
(das Szenario von PR #43: dort `a` 138 GiB, dann Exit 137). `a`: 262144 × 64, Headroom 2836 MiB (nach der Formel in
`static-streaming.md`: `b` bleiben 1,96 GiB claimable für 166 MiB Cache + die Hälfte der Statik von 3,58 GiB), `b`: 4096 × 1,
Headroom 96 MiB. `b` streamt tatsächlich (`static_streamed_fraction` 0,562). Host-RAM 170 GiB (MemAvailable im Leerlauf 161 GiB).

| Phase | Budget | `a` aktiv | Anfrage an `b` | danach |
|---|---|---|---|---|
| `natural` | 0 (aus MemAvailable/cgroup) | Pool 73,3 GiB + Store 67,5 GiB = 140,8 GiB gebucht | **200**; Store-Kopie von `a` (67,5 GiB, aktives Model) verdrängt, Pools 139,4 GiB; `b` aktiv, 56,2 % gestreamt | `a` aktiv, gleiche Antwort wie vorher, Server lebt, VRAM danach 57 MiB je Karte |
| `refuse` | `a`-Pools + 256 MiB (73,6 GiB) | Pool 73,3 GiB | **503** „activation of model b failed: not enough host memory (RAM, cgroup limit or host budget)“; Log: `host budget: this model's host pools (65.75 GiB) do not fit cache.host_budget_mib (73.60 GiB) after every prepared model and every store copy was dropped. Nothing running is touched` | `b` `failed`, `over_budget_claims` 3, `a` aktiv, gleiche Antwort, Server lebt |

Host-RAM während des Laufs (`free -m` und RSS alle 4 s, Phase `natural`): Spitze 154,6 GiB „used“ (RSS 144,7 GiB), kleinstes
MemAvailable **16,3 GiB** (≥ `--host-headroom-mib` 4 GiB), nie unter dem Headroom; nach der Deaktivierung zurück auf 10 GiB.
Alle Prüfungen des Skripts in beiden Phasen bestanden, kein „Killed“ im Log. `ctest -L unit` auf dem Merge-Stand: nur
die acht bekannten roten Tests (arch, gbnf shard0/2, tokenizer, images, videos, kbench, kbench_mutant).

Fehler: keine neuen. Ohne den Fix hätte dieselbe Aktivierung Store-Kopie (67,5 GiB) + Pools beider Models (≈ 140 GiB)
gebucht, also ≈ 208 GiB auf 170 GiB (der Kernel-Kill aus §10); hier verdrängt die Prüfung zuerst die Kopie und passt.
Nicht gemessen: ein Lauf desselben Szenarios auf `main` zur Gegenprobe (die Kill-Gefahr für den Container ist es nicht
wert), und `b` mit `--ngram-placement ram`.

## 14 Erschöpfte VA-Arena und Single-Engine-Perf (#45)

Design und Ursache: `elastic-slab.md`, Abschnitt „Erschöpfte Arena (#45)". Kurz: Bei erschöpfter Arena schrumpft der
Slab nur noch logisch (Seiten bleiben gemappt, Warnung einmal, Gauge `radiance:model_expert_slab_arena_spent`); die
stehengebliebenen Demote-Kopien kamen aus `Mover::resize`, das Moves im Zustand `Published`/`Quarantine` ein
zweites Mal zählte (Fill promotete, das nächste `resize` demotete, endlos). Tests: Host (`ROCR_VISIBLE_DEVICES=99`) und
Karte (`place_test` ohne die Variable) grün; ohne den Fix hat der Test zehntausende Demotionen.

**Elastisch auf der Karte (Regression).** `m_el.json` (zwei MoE, 26 000 MiB Host-Pool), `elastic.py m1:45,idle:15,m2:60,idle:20`,
je zwei Läufe auf dem Stand dieses Branches und auf `origin/main` (ohne den Fix, abwechselnd): Greedy-Tokens nach
jeder Phase gleich der Referenz (4 von 4 Phasen, alle vier Läufe), keine Fehler, VRAM zwischen 25,5 und 30,6 GiB,
am Ende 57 MiB je Karte. Die Prüfung „Summe der `committed_bytes` ≤ Pool“ meldet in allen Läufen vereinzelt
„False“ (main: 1 und 2 von 20 Samples, Branch: 1 und 3 von 20, bis zu 864 MiB über dem Pool): das ist die Lesung
einer nicht atomaren Meldung, kein neuer Befund, und gleich auf `main`.

**Single-Engine ohne Manifest gegen den Stand vor #44.** Gebaut: `d66fc89` (Merge von PR #42, der Commit vor dem
Merge von PR #44 auf `main`) gegen diesen Branch (`origin/main` + #45). `radiance --model … --tp 2`, Port 18080,
je Lauf ein frischer Server; die Aufbauten abwechselnd (vor, nach, vor, nach), `bench.py` mit je 3 Wiederholungen
(C1: 256 Token; C8: acht Requests; Prefill: ein Request, tok/s aus der Antwort). Zahlen: Mittel über die Läufe,
in Klammern Minimum–Maximum.

| Model, Optionen | Messgröße | vor #44 | dieser Branch | Δ |
|---|---|---|---|---|
| dicht Qwen3.5-0.8B, `--max-model-len 8192 --max-num-seqs 8` (2 Läufe × 3) | C1 tok/s | 328,9 (325,3–332,8) | 328,0 (324,3–333,2) | −0,3 % |
| | C8 tok/s | 2047 (2040–2059) | 2042 (2022–2052) | −0,3 % |
| | Prefill tok/s (~4300 Token) | 65 213 (60 008–67 021) | 64 974 (60 574–66 686) | −0,4 % |
| MoE Qwen3.8-Flash-Next, `--max-model-len 4096 --max-num-seqs 8 --placement expert_tiered --host-pool-mib 26000 --kv-cache-dtype fp8 --tp-wire wht6 --num-speculative-tokens 3 --max-num-batched-tokens 1024 --ngram-placement disk` (2 Läufe × 3) | C1 tok/s | 172,3 (147,7–189,8) | 172,2 (148,1–188,9) | −0,1 % |
| | C8 tok/s | 599 (474–806) | 625 (472–802) | +4,4 % (Rauschen) |
| | Prefill tok/s (~3,3k Token) | 4443 (4210–4728) | 4297 (4028–4891) | −3,3 % (Rauschen) |
| MoE, Prefill allein, 2 Läufe × 8 Wiederholungen | Median tok/s | 4326 (Lauf 1: 4462, Lauf 2: 4084) | 4388 (Lauf 1: 4417, Lauf 2: 4102) | +1,5 % (Läufe −1,0 % / +0,4 %) |

Dicht: alle drei Größen innerhalb 0,5 %, das Ziel ±2 % ist erfüllt. MoE C1: 0,1 %, und die drei Wiederholungen
steigen in beiden Aufbauten gleich (148 → 180 → 189 tok/s, die Heat wärmt sich). MoE C8 und Prefill streuen
von Lauf zu Lauf stärker als zwischen den Aufbauten (C8 474–806 tok/s, Prefill 3880–5196 tok/s auch innerhalb eines
Aufbaus, zweigipflig je nach Heat-Zustand); der erste Eindruck eines Prefill-Rückstands von 13 % in der ersten
Wiederholung (4118/4028 gegen 4728/4715) verschwand bei 2 × 8 Wiederholungen (Mediane ±1,5 %). Eine Abweichung
über ±2 % lässt sich bei MoE-C8 und -Prefill mit dieser Streuung weder ausschließen noch belegen; ein systematischer
Unterschied war in keiner Messreihe zu sehen. Der Pfad ohne Manifest nutzt weder den elastischen Pool noch den
geänderten `resize` (kein Cap), die Änderung dieses Tickets berührt ihn nicht.

Nicht gemessen: drei oder mehr Models, die Arena wirklich erschöpft auf der Karte (zu viele Stückwechsel:
65 000 je Rank; auf der Karte nur im `place_test` mit kleiner Arena).

Beobachtung, nicht behoben (schon auf `main`): `place_test_card` (`ctest -R place_test_card`, echte Karte) endet im
Test `a_promoted_static_unit_is_copied_to_fresh_arena_address_in_keep_order_within_the_cap` mit SIGSEGV, weil er den
Inhalt von VRAM mit einem Host-Zeiger liest (`place_test.cpp`, `r.at(…)[i]`); mit `ROCR_VISIBLE_DEVICES=99` (Host-Backend) läuft
er. Alle anderen Fälle laufen auf der Karte.

## 15 Die 13-%-Lücke: MoE im Manifest mit zwei Models (C1)

`scripts/swap-acceptance/gap13.py OUTDIR 5 6 VARIANTEN`: je Runde ein frischer Server je Variante (die Reihenfolge der
Varianten rotiert von Runde zu Runde), Aufwärmen mit 4 (statisch) bzw. 10 (dynamisch) C1-Requests, dann 6 C1-Requests
(`Write a long story about a robot.`, 256 Tokens, Greedy, `ignore_eos`), 5 Runden, also 30 Messwerte je Variante.
Das MoE ist das von `m_dm.json`, TP=2, Port 18080, GPUs exklusiv; in den Manifest-Varianten steht daneben das
dichte `d` (0,8B), das nie aktiviert wird (kalt).

**Statischer Aufbau** (wie §8: `--deterministic`, `--gpu-headroom-mib 5000` im Model; die Heat ist aus):

| Variante | C1 tok/s Mittel (Median, Min–Max) | gegen `--model` |
|---|---|---|
| `--model` | 90,8 (91,0; 89,7–91,6) | – |
| Manifest, nur das MoE (`m_gap1.json`) | 90,9 (91,0; 90,1–91,4) | +0,1 % |
| Manifest, MoE + kaltes `d` (`m_dm.json`) | 90,5 (90,4; 89,9–91,3) | −0,3 % |
| dasselbe mit `RADIANCE_ELASTIC_SLAB=off` (einfacher Pool) | 90,8 (91,0; 89,8–91,3) | 0,0 % |
| MoE + `d`, `d` einmal aktiviert und wieder deaktiviert (MoE blieb aktiv) | 90,3 (89,8; 89,3–91,4) | −0,5 % |

Die Residenz ist in allen Varianten gleich (`/stats`: 18 185 Einheiten im VRAM, 6893 gepinnt). **Die Lücke von 13 %
aus §8 (74,8 → 65,2 tok/s) ist auf dem heutigen Stand nicht mehr zu sehen**; die Werte liegen ohnehin höher (91 statt
72–75 tok/s für `--model` mit Headroom 5000), das alte Bild entstand also in einem anderen Zustand des Hosts oder des
Builds. Was es war, ist nicht geklärt. Weder der elastische Pool (Variante ohne ihn), noch die Tape-Umgehung durch
eine zweite registrierte Engine (`d` kalt bzw. nach der Deaktivierung), noch der GroupPlanner-Tick (die Heat ist hier
aus, der Tick meldet nur die Slab-Größe) kosten messbar etwas.

**Dynamischer Aufbau** (ohne `--deterministic` und ohne `--gpu-headroom-mib` im Model, `--host-pool-mib 20000`;
Heat an): hier gibt es die Lücke, und sie ist der VRAM-Anteil.

| Variante | C1 tok/s Mittel (Median, Min–Max) | gegen `--model` | Experts im VRAM (`routed_res`) | Einheiten VRAM / gepinnt | Slab je Rank |
|---|---|---|---|---|---|
| `--model` | 186,8 (186,7; 184,7–189,1) | – | 99,51 % | 22 188 / 2890 | – |
| Manifest, nur das MoE | 186,8 (186,5; 184,7–189,7) | 0,0 % | 99,51 % | 22 188 / 2890 | 26 319 MiB |
| Manifest, MoE + kaltes `d` | 169,6 (169,7; 168,5–170,5) | **−9,2 %** | 95,41 % | 8778 / 16 300 | 10 375 / 10 422 MiB |

Ursache belegt: das Manifest rechnet den Anteil jedes Models als `(frei − Headroom) / n` (`manifest_share_mib`,
`core/main.cpp`), auch für Models, die nie aktiv werden. Im Log: `manifest: model 'moe' takes about 16235 MiB … 
(--gpu-headroom-mib 16331)` gegen den Standard-Headroom 96 MiB bei einem Model. Das halbierte Expert-Budget lässt
nur 8778 statt 22 188 Einheiten im VRAM (Rest gepinnt), `routed_res` fällt von 99,5 auf 95,4 %, und C1 fällt um 9,2 %
(bei 13 % liegt die Lücke nicht; sie hängt vom Anteil der nicht residenten Experts ab). Im statischen Aufbau greift der
Anteil nie, weil `--gpu-headroom-mib` im Model steht und den Anteil ersetzt.

**Nicht behoben.** „Keinen Anteil für nicht aktive Models reservieren“ ist nicht klein: die Aktivierung des zweiten
Models braucht die Bytes gleich bei der Aktivierung (Statik, KV, Arena, Staging), und nur die Experts kann der Planer
später zurückholen; ein erstes Model mit der ganzen Karte würde die zweite Aktivierung in den Druckpfad
(`RAD_E_NOMEM`, Deaktivierung des ersten) schicken. **Empfohlenes Folge-Issue:** „Manifest: Anteil des ersten
Models wächst auf die ganze Karte, solange die anderen kalt sind, und schrumpft über den GroupPlanner, wenn eines
aktiv wird“ (Expert-Slab elastisch, Statik/KV/Arena des ersten Models auf den Anteil begrenzt). Bis dahin: ein
eigenes `--gpu-headroom-mib` je Model (wie in `m_dm.json`) oder `cache.max_active_models: 1` (nach dem Code ist der
Anteil dann die ganze Karte; nicht gemessen, die zweite Aktivierung deaktiviert das erste Model).

## 16 GroupPlanner unter Dauerlast mit wechselnder Gewichtung und die Summe der Commits (#51)

Zwei MoE-Einträge (`m_el.json`, TP=2, `idle_timeout_ms` 0), `elastic.py LOG m1:40,idle:15,w80/20:90,w20/80:90,w50/50:60,w80/20:60,w20/80:60,m2:40`.
`wA/B` belastet beide Models zugleich mit Dauerlast: 8 Client-Threads für `m1` und 2 für `m2` bei `w80/20`, umgekehrt bei
`w20/80`, 5 und 5 bei `w50/50`. Jeder Request fragt `logprobs` (ein nicht endliches Logit wäre ein HTTP 500 und die
Logzeile `no finite logit`); nach jeder Phase die Greedy-Tokens beider Models gegen die Referenz (`5e8678d0682f`);
`/radiance/models` wird alle 100 ms gelesen und gegen den Pool geprüft. Der Pool ist je Rank die Summe der Angebote
(Rank 0: 10 314 + 5 124 = 15 438 MiB; `m2` tritt mit 5,08 GiB bei, weil die Karte nach `m1` nicht mehr frei hatte,
darum ist der Pool kleiner als in §9). Spalten: `committed / target` in MiB, Rank 0, am Ende der Phase.

| Phase | m1 | m2 | Σ committed ≤ 15 438 | Samples / Überschreitungen | Tokens / Fehler | Greedy = Referenz | `no finite logit` |
|---|---|---|---|---|---|---|---|
| `m1:40` | 9 303 / 9 304 | 6 135 / 4 451 | 15 438 | 701 / 0 | 512 / 0 | ja | 0 |
| `idle:15` (Heat kühlt) | 6 854 / 4 451 | 8 583 / 8 584 | 15 437 | 371 / 0 | 0 / 0 | ja | 0 |
| `w80/20:90` | 9 309 / 9 309 | 6 130 / 4 451 | 15 439 (+1) | 1 294 / 0 | 384 / 0 | ja | 0 |
| `w20/80:90` | 6 854 / 4 451 | 8 583 / 8 584 | 15 437 | 1 279 / 0 | 384 / 0 | ja | 0 |
| `w50/50:60` | 7 446 / 5 822 | 7 993 / 7 993 | 15 439 (+1) | 1 051 / 0 | 384 / 0 | ja | 0 |
| `w80/20:60` | 10 239 / 10 240 | 5 199 / 4 451 | 15 438 | 979 / 0 | 256 / 0 | ja | 0 |
| `w20/80:60` | 6 854 / 4 451 | 8 583 / 8 584 | 15 437 | 955 / 0 | 256 / 0 | ja | 0 |
| `m2:40` | 6 854 / 4 451 | 8 583 / 8 584 | 15 437 | 702 / 0 | 513 / 0 | ja | 0 |
| **gesamt** | | | | **7 736 / 0** | **2 689 / 0** | 8 von 8 | **0** |

- Die Targets ändern sich mit der Gewichtung: bei `w80/20` bekommt `m1` rund 9,3–10,2 GiB und `m2` fällt auf sein Minimum
  (Target 4 451), bei `w20/80` kehrt es sich um (`m2` 8 583, `m1` 4 451), bei `w50/50` liegen beide bei 5,8 / 8,0 GiB.
  Das Target des Gebers fällt sofort, das des Nehmenden folgt der Freigabe; `committed` folgt dem Target in Schritten
  von 128–510 MiB je Meldung (100 ms), ein voller Wechsel von 3 GiB dauert 10–20 s. Danach bleiben die Werte stehen
  (`m2` unter 20 % Last gibt nicht alles ab: 6 130 gegen 4 451 Target ist der Rest, den der Geber nicht freigeben kann,
  weil sein Heat-Rest belegt ist).
- VRAM je Karte bleibt zwischen 26,9 und 27,3 GiB (Basis mit zwei Models 27 290 MiB); am Ende des Laufs, nach dem
  Beenden des Servers, 57 MiB je Karte (Grundlinie).
- Keine nicht endlichen Logits, kein Hänger, kein HTTP-Fehler in 8 Phasen mit `logprobs` bei jedem Request. Das in §9
  beschriebene „Bekannte Risiko“ (`row 0 has no finite logit`) trat in diesem Lauf nicht auf; `lpstress.py` wurde nicht
  wiederholt, das Risiko ist damit nicht ausgeschlossen. Die Phasen sind 60–90 s lang; eine Dauerlast über Stunden ist
  nicht gemessen.

### #51: Summe der `committed_bytes` über dem Pool: echte Überbuchung, behoben

Gleicher Lauf (`m1:40,idle:15,w80/20:60,w20/80:60,m2:40,m1:40`, Samples alle 100 ms), erst auf `origin/main`, dann
jeweils mit einem weiteren Fix; „Überschreitung“ heißt Σ `committed` > Σ Angebote (ohne Toleranz):

| Stand | Samples | Überschreitungen | größte Überschreitung | VRAM Karte 0 Spitze |
|---|---|---|---|---|
| `origin/main` | 4793 | 3561 (74 %) | 3 399 MiB | 30 396 MiB (+3,1 GiB über der Basis) |
| `main` + ein Lesen aller Models aus einem Planer-Schnappschuss | 4802 | 3471 (72 %) | 3 398 MiB | – |
| Branch: Beitritt hält sein ganzes Angebot, gekürzte Grants bleiben bis zur Meldung (WIP-Stand) | 4791 | 3603 | 3 329 MiB | 28 208 MiB |
| + kein Grant über den verkleinerten Pool | 4718 | 3561 | 3 407 MiB | 28 156 MiB |
| + Wachstum der Klassen höchstens um den gewährten Betrag | 4836 | 572 | 1 MiB | 27 292 MiB (Basis) |
| endgültiger Stand, Toleranz 2 MiB (Rundung), 8 Phasen (oben) | 7 736 | 0 | 0 | 27 292 MiB (Basis) |

- **Kein Messartefakt.** Das Lesen aus einem Schnappschuss (`GroupPlanner::view_all`; vorher las der Server jedes Model
  mit einer eigenen Sperre) ändert die Zahl der Überschreitungen nicht (74 % → 72 %), und der VRAM der Karte stieg
  tatsächlich um 3,1 GiB über die Basis: `m1` hatte Seiten gemappt, die `m2` noch nicht zurückgegeben hatte. Die Karte
  hatte den Platz noch frei, darum fiel nichts aus; bei vollem VRAM wäre der Commit gescheitert.
- **Ursachen** (jede mit Test): (1) Ein beitretendes Model galt bis zu seiner ersten Meldung als leer, der Planer gab
  die Bytes zwischen seinem Start und seinem Angebot dem laufenden Model, die das Neue dann zugleich beanspruchte
  (`a_joining_model_holds_its_whole_offer_until_it_reports`). (2) Ein Grant, der gekürzt wurde, bevor der Nehmende ihn
  benutzt hatte, ging sofort an ein anderes Model, während der Mover des Nehmenden weiter bis zum alten Target mappte
  (`a_grant_that_is_cut_stays_reserved_until_the_taker_reports`; dazu nimmt der Engine-Tick jetzt jedes Absenken des
  Caps sofort, nicht erst ab 1 %). (3) Ein Beitretender, der mit weniger startet als er anbot, verkleinert mit seiner
  ersten Meldung den Pool; sein Grant aus dem größeren Pool blieb als „noch nicht benutzt“ reserviert
  (`a_newcomer_that_starts_with_less_than_it_offered_is_not_granted_past_the_smaller_pool`; Fix in `group_gate`: was der
  Pool nicht mehr deckt, kommt von den nicht gedeckten Grants zurück). (4) Der Planer gewährt Bytes für den ganzen
  Slab, der Mover wächst je Slab-Klasse (zwei Klassen mit 0,97 und 1,45 MiB je Slot): in den Logs wuchs `m1` um 128 MiB
  je Meldung von 6 988 auf 10 220 MiB, während sein Target gleich `held` war und `m2` nicht freigab. Dass die
  Zusammensetzung der Klassen der Auslöser ist, folgt aus der Wirkung des Fixes (aus 3561 wurden 572 Überschreitungen
  von höchstens 1 MiB), nicht aus einer Einzelmessung des Mover-Zustands. Fix: `group_limit_growth` begrenzt die Caps
  über dem, was die Klassen halten, zusammen auf die gewährten Bytes (`class_caps_grow_together_by_no_more_than_the_granted_bytes`).
  Dazu ein Simulationstest (`movers_that_follow_their_targets_never_hold_more_than_the_pool`: 2 und 3 Models, 200 Startwerte,
  400 Schritte, zufällige Heat); er fand auf `origin/main` die Überbuchung (`a` hält 30, `b` 12, Pool 40).
- **Der Rest ist Rundung.** Die vier Zahlen je Rank werden einzeln auf ganze MiB gekürzt, die Summe kann bis 2 MiB über
  dem Pool lesen (größter Wert 1 MiB, die Zeilen „+1“ oben). Die Prüfung in `elastic.py` und `lpstress.py` hat darum
  `POOL_SLACK_MIB = 2` und zählt Samples, Überschreitungen und die größte Überschreitung je Phase und gesamt.

## 17 TTFT getrennt nach Queue, Aktivierung und Prefill, je Request (#21)

Jede Antwort trägt in `timings` jetzt `activation_ms` (Wartezeit im `ModelRouter`, bis das Model aktiv war; 0, wenn es
das schon war) und `queue_ms` (vom Einreichen beim Scheduler bis zur Admission); `prompt_ms` ist der Prefill (bis zum
ersten Token), `ttft_ms` die Summe samt Parsen und Tokenisieren. Dieselben drei Teile als Histogramme je Model in
`/metrics`: `radiance:model_ttft_activation_seconds`, `radiance:model_ttft_queue_seconds`,
`radiance:model_ttft_prefill_seconds` (Label `model`, die Buckets von `vllm:time_to_first_token_seconds`; jeder Request
mit erstem Token zählt in allen dreien, auch mit Aktivierungsanteil 0). Host-Tests:
`model_activation_test: a_request_that_activates_its_model_reports_the_wait_as_activation_and_a_warm_one_zero`,
`the_scheduler_queue_wait_is_reported_apart_from_the_activation`.

Aufbau: `m_ttft.json` (zwei Qwen3.5-0.8B `a` (Standard, 4096 Token, 8 Seqs) und `b` (2048 Token, fp8-KV, 4 Seqs),
`idle_timeout_ms` 3000, `--deterministic`), TP=2; `ttft_run.sh 6` (`ttft.py`) startet sechs frische Server. Jeder
Request: Completion mit ~1700 Prompt-Tokens (zufälliger Text, kein Prefix-Treffer), 4 Token (unter Last 64), Werte aus
`timings`. Zahlen in ms, p50 / p95 über alle Requests des Falls.

| Fall | n | Queue | Aktivierung | Prefill | TTFT |
|---|---|---|---|---|---|
| (a) Model aktiv, ein Request nach dem anderen | 90 | 0,1 / 0,2 | 0 / 0 | 23,7 / 25,0 | 24,0 / 25,5 |
| (b) Model kalt (erster Request an `b` im frischen Prozess, `a` aktiv) | 6 | 0,5 / 0,8 | 1968 / 2007 | 37,9 / 38,1 | 2008 / 2045 |
| (c) Model vorbereitet (warm; `a` und `b` abwechselnd, vorher alle deaktiviert) | 48 | 0,4 / 0,8 | 2125 / 2227 | 28,8 / 29,5 | 2155 / 2256 |
| (d) unter Last: 32 Clients auf `a` (8 Seqs), 192 Requests | 192 | 1403 / 1461 | 0 / 0 | 27,1 / 103,6 | 1431 / 1490 |
| (d′) dieselbe Last auf `a`, dazu ein kalter Request an `b` | 5 (`b`) | 0,4 / 0,6 | 2053 / 2120 | 29,9 / 30,3 | 2083 / 2151 |
| (d′) die Requests der Last auf `a` währenddessen | 320 | 1375 / 1480 | 0 / 0 | – | – |

- **Die Teile summieren sich.** TTFT − (Queue + Aktivierung + Prefill) ist der Rest aus Parsen, Template und
  Tokenisieren: Median 0,2–0,4 ms je Fall; Maximum 0,6 ms bei kalt und warm, 2,5–4,0 ms bei aktiv und unter Last (einzelne Ausreißer). Die Perzentile oben stehen
  einzeln; sie summieren sich nur ungefähr.
- **Aktivierung bestimmt den TTFT eines Models, das nicht läuft** (rund 98 %); der Prefill (1700 Token, 24–38 ms) und
  die Queue (unter 1 ms) spielen dort keine Rolle. Kalt war hier nicht langsamer als warm (1968 gegen 2125 ms): das
  0,8B-Model liegt im Page Cache, und die Aktivierung besteht fast nur aus dem Aufbau der Engine, nicht aus dem Lesen
  der Gewichte (`weights` 0,25 s warm, §2). Der Unterschied kalt/warm der Gewichte ist beim MoE groß (§8: 70,2 s gegen
  60,0 s; dort nicht erneut gemessen, eine Aktivierung dauert eine Minute, die Aufteilung dort bleibt Aktivierung ≫ 99 %).
- **Unter Last bestimmt die Queue den TTFT.** 32 Clients bei 8 Sequenzen: 1,4 s Warten auf die Admission gegen 27 ms
  Prefill (p95 104 ms, weil die Prefills mit dem Decode der anderen Sequenzen teilen). Die Aktivierung eines zweiten
  Models unter dieser Last dauerte nicht länger als ohne (2053 gegen 1968 ms im p50), und die Queue von `b` blieb
  unter 1 ms: seine Warteschlange ist die eigene, nicht die von `a`.
- **Der erste Prefill nach einer Aktivierung ist langsamer** (38 ms im Fall b, 29 ms im Fall c, 24 ms im Fall a):
  nicht untersucht, wahrscheinlich Kernel-Aufwärmen.
- **Quervergleich mit den Metriken.** `/metrics` nach einem Lauf (Fall a bis d′ in einem Prozess):
  `radiance:model_ttft_activation_seconds_count` je Model gleich der Zahl der Requests an das Model (198 für `a`, 22
  für `b`), `_sum` für `b` 11,95 s (1 kalte, 4 warme und 1 Aktivierung unter Last), für `a` 10,3 s (fünf warme).
- **Nicht gemessen:** MoE (siehe oben), Queue beim Streaming (der Beobachter hängt in `Server::record()`, das auch der
  Stream-Pfad am Ende ruft; gemessen wurden nur Nicht-Stream-Antworten), Requests mit `n > 1`
  (Queue vom frühesten Einreichen bis zur frühesten Admission, Prefill bis zum letzten ersten Token).
