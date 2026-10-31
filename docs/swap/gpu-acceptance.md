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
| 8 | Durchsatz / TTFT | gemessen; zwei gleichzeitig aktive Models bremsten sich stark, nach Fix 2 auf rund die Hälfte statt ein Zehntel; TTFT-Aufteilung nur abgeleitet (keine Metrik) |
| – | #13 Host-Budget mit großem Model | bestanden nach Fix 3; Host-RAM ist dabei der Engpass (siehe unten) |
| – | Issue-Kriterien Sampling, MTP, Streaming, Abbruch, lange Prompts | bestanden |

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

**TTFT nach Queue / Aktivierung / Prefill.** Es gibt keine Metrik je Request für Queue und
Aktivierung. Abgeleitet: Request-Wanduhr = Router-Wartezeit + `activation_seconds` (aus
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
