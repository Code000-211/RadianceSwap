# Zero-Copy-Streaming statischer Gewichte (#18)

Ein Model wird auch aktiv, wenn seine statischen (Nicht-Expert-)Gewichte nicht in den freien VRAM
passen, der ihm bleibt. Was nicht auf die Karte passt, liegt in gepinntem Host-Speicher (dem Pool,
in dem heute die Experts liegen, die nicht passen) und wird im Decode per Zero-Copy gelesen; im
Prefill kopiert der Stager-Ring (`core/place/stager.h`) es vorab auf die Karte. Die Bytes sind
dieselben, die Ausgabe ist bit-identisch. Das ist nach ADR 0002 nur ein Übergang: PCIe liefert
≈ 28 GB/s gegen 600–870 GB/s aus dem VRAM (`measure-static-streaming.md`), der Decode bricht um
den gestreamten Anteil ein. Wer die Gewichte danach in den VRAM zurückholt, ist Ticket #19.

Ohne Bedarf, also wenn die Statik neben dem Cache einer Sequenz passt, ändert sich nichts.

## Was sich ändert

| Stelle | Änderung |
|---|---|
| `core/mem/vram_budget.cpp` | Passt die Statik nicht neben einen brauchbaren Cache (`kv_one`, eine Sequenz mit `--max-model-len`, höchstens 1 GiB: `kStreamCacheMax`, ein fester Wert, kein Wert je Token) und ist die Platzierung `auto` oder `expert_tiered`, bekommt die Karte, was nach diesem Cache und dem Ring übrig ist (ganze MiB); der Rest ist `statics_streamed`. Der Cache behält diese Größe, die Experts bekommen nichts. Passt die Statik neben diesen Cache, bleibt alles wie es war, auch wenn der Cache dann weniger als eine ganze Sequenz hat: wegen eines langen `--max-model-len` wird nichts gestreamt. Vorher bekam der Cache, was nach der ganzen Statik blieb, auch weniger als eine Sequenz. `all_vram` und `layer_offload` behalten den alten Boden. |
| `core/place/planner.*` | `assign_static_stream`: die statischen Units, die passen, bleiben auf der Karte, der Rest geht in den gepinnten Pool auf `Site::DeviceZeroCopy` (kein Slab-Slot, kein Staging-Ring, keine Prefetch-Liste, kein Host-Site). Es ist kein zweiter Placement-Algorithmus: dieselben Tier/Site-Achsen, derselbe `HostFall`. `auto` wählt es für ein Model ohne Experts (`Strategy::StaticStream`, nur über `auto` erreichbar); `expert_tiered` ruft es dort auf, wo es früher `assign_layer_offload` aufrief. Ein ausdrückliches `layer_offload` bleibt unverändert. |
| `core/place/stager.cpp` | Eine Schicht im Ring besteht aus ihren Expert-Units und ihren gestreamten statischen Units. Modellweite Units (der Ausgabekopf) haben keine Schicht und werden an Ort und Stelle gelesen: ein Prefill rechnet sie nur für die letzten Zeilen. |
| `core/engine_bringup.cpp` | Der Host-Pool wächst vor dem Split um das, was ihm fehlt (gestreamte Statik, dazu alle Experts, für die die Karte keinen Platz hat), wie schon für den Drafter. Ohne kleinere Arena-Stufe (jede Architektur außer `qwen4exp_fp8`/`glm5next`) wird der Ring nicht verliehen, gehört dem Stager und wird ab 256 Zeilen (wie die kleinste Stufe) scharf geschaltet. |
| Status | `WeightLoadReport::static_bytes`/`static_streamed_bytes` (Plan, über alle Ranks); `/radiance/models` zeigt `static_bytes`, `static_streamed_bytes`, `static_streamed_fraction`, `/metrics` `radiance:model_static_bytes`, `…_streamed_bytes`, `…_streamed_fraction` (nur für aktive Models). Der Startbericht nennt den Anteil, `Plan::report` die Zeile `static stream`. |

## Welche Statik auf die Karte kommt

Zuerst die Karte, dann der Pool; was gebraucht wird, bleibt auf der Karte. Der Schlüssel kommt nur
aus dem Model (jeder Rank entscheidet gleich):

1. die Gewichte des Drafters (`mtp.`; sie laufen in jedem Draft-Pass),
2. modellweite Gewichte, die jedes Token liest (der Ausgabekopf: ein Prefill rechnet ihn nur für
   wenige Zeilen, der Stager deckt ihn nicht ab),
3. Schicht-Units, in Ausführungsreihenfolge (die zuletzt laufenden gehen zuerst in den Pool),
4. Gewichte, die nur je Request gelesen werden (Encoder-Türme),
5. zuletzt eine Token-Embedding-Tabelle (ein Lookup weniger Zeilen je Schritt).

Eine Unit, die nicht mehr passt, wird übersprungen; kleinere dahinter füllen die Karte weiter.

## Der Ring im Prefill

Er kommt aus der Statik, die die Karte nicht hält: zwei Schichten mal ihr gestreamter Anteil
(`static_layer_max`), in wenigen Schritten bestimmt, weil der Ring selbst die Karte verkleinert.
Die Einheit, die der Planner zuletzt streamt, ist eine ganze Unit, daher liegt der echte Anteil bis
zu einer Unit über dem, den der Split meldet; ein Ring, der für eine Schicht zu klein ist, stagt,
was passt, und liest den Rest an Ort und Stelle (wie bei Experts). Unter `--deterministic` gibt es
keinen Stager (der Plan ist dort statisch): der Prefill liest direkt, was ADR 0002 für den
Normalfall ausschließt.

## Host-Pool

`--host-pool-mib` wird vor dem Split um `statics_streamed + experts + größte statische Unit + 8 MiB`
abzüglich der Größe des Pools erhöht, wenn das nötig ist (Log: „--host-pool-mib grows by N“). Ein
MoE mit wenig VRAM-Anteil braucht damit alle Experts im Pool (die Karte hat für keinen Platz);
das Host-Budget (#13) zählt die Vergrößerung als Pool.

## Tests (Host, `ROCR_VISIBLE_DEVICES=99`)

- `place_test`: die Statik, die nicht passt, wird host-resident auf `DeviceZeroCopy` geplant (ohne
  Slab, Prefetch, Host-Run); die Reihenfolge (Drafter, Token-Gewichte, Schichten, Türme,
  Embedding) und „überspringen statt abbrechen"; Experts und gestreamte Statik teilen Karte und
  Pool; der Pool zu klein → `unfit` mit gefülltem Plan; passt alles → nichts gestreamt, und
  `layer_offload` staged weiter; der Stager kopiert gestreamte Schichten voraus und gibt sie
  zurück, den Kopf nicht.
- `mem_test`: der Split bei Statik über dem Boden (Karte, Ring, Cache = eine Sequenz, ganze MiB),
  ohne Stufe kein Ring, passende Statik und nicht streamende Platzierungen unverändert.
- `engine_e2e_test`: `g31` (libref) und `g31` (libavx) mit zu kleinem Gerät
  (`RADIANCE_HOST_VRAM_MIB`, gesucht, nicht fest): ein Teil der Statik kommt aus dem Pool, der Ring
  ist scharf (Prompts über 256 Zeilen), und die Aufnahme (KL-Modus) ist Byte für Byte die des Laufs
  mit allem auf der Karte.
- `model_deactivation_test`: Anteil in Status und Metriken, nur für aktive Models.

## GPU-Abnahme

`scripts/swap-acceptance/static_streaming.py MODEL_DIR [--pct 50] [--tp 2] [--phases ref,stream,status,full]`
(Port 18080, braucht die GPUs für sich). Phasen: `ref` (Basis: Tokens und Durchsatz mit allem auf
der Karte, daraus die Zahlen des Budgets), `stream` (Headroom so, dass `pct` % der Statik
streamen, Tokens müssen gleich sein, `bench.py` zeigt die Kosten), `status` (Manifest, Status und
Metriken zeigen den Anteil), `full` (Model `a` hält den VRAM, `b` aktiviert sich bei vollem VRAM,
streamt, antwortet gleich, `a` läuft weiter; Aktivierungszeit und erste Antwortzeit von `b`).
Die Herleitung der Headrooms steht im Skript; `m_ss.json` ist das Manifest der Phase `full` zum
Von-Hand-Start (`HA` ersetzen). Für `b` im Zustand `full`: `claimable_b = claimable_0 - used_a` mit
`used_a = F - 16 MiB - H_a` (F die freien MiB der Basis), also `H_a = F - 16 - claimable_0 + K + (1-pct)·S`
(K der Cache einer Sequenz, S die Statik, alles in MiB aus dem Startbericht der Basis).
