# Zero-Copy-Streaming statischer Gewichte (#18)

Ein Model wird auch aktiv, wenn seine statischen (Nicht-Expert-)Gewichte nicht in den freien VRAM
passen, der ihm bleibt. Was nicht auf die Karte passt, liegt in gepinntem Host-Speicher (dem Pool,
in dem heute die Experts liegen, die nicht passen) und wird im Decode per Zero-Copy gelesen; im
Prefill kopiert der Stager-Ring (`core/place/stager.h`) es vorab auf die Karte. Die Bytes sind
dieselben, die Ausgabe ist bit-identisch. Das ist nach ADR 0002 nur ein Übergang: PCIe liefert
≈ 28 GB/s gegen 600–870 GB/s aus dem VRAM (`measure-static-streaming.md`), der Decode bricht um
den gestreamten Anteil ein. Wer die Gewichte danach in den VRAM zurückholt, ist Ticket #19 (Abschnitt
„Zurück in den VRAM“ unten).

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
keinen Stager (der Plan ist dort statisch): der Prefill läse direkt über PCIe, was ADR 0002
ausschließt. Deshalb lehnt `Engine::plan` dort ein Streaming statischer Gewichte ab (Startfehler
„--deterministic: N of the static weights do not fit … has no prefill stager“, Exit 1; Test
`deterministic_refuses_streaming_statics`). Passt die Statik neben den Cache, ändert sich nichts.

## Host-Pool

`--host-pool-mib` wird vor dem Split um `statics_streamed + experts + größte statische Unit + 8 MiB`
abzüglich der Größe des Pools erhöht, wenn das nötig ist (Log: „--host-pool-mib grows by N“). Ein
MoE mit wenig VRAM-Anteil braucht damit alle Experts im Pool (die Karte hat für keinen Platz);
das Host-Budget (#13) zählt die Vergrößerung als Pool. Vor der Allokation prüft `Engine::admit_host_pools` den ganzen Pool gegen MemAvailable, cgroups und Budget (verdrängt Store-Kopien, sonst `RAD_E_NOHOST`, 503; #46, `host-budget.md`).
GPU-Nachweis: `scripts/swap-acceptance/host_budget_stream.py`.

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

`scripts/swap-acceptance/static_streaming.py MODEL_DIR [--pct 50] [--tp 2] [--phases ref,stream,status,full] [--seqs 8] [--extra "ARGS"] [--hog PATH --hog-extra "ARGS"]`
(Port 18080, braucht die GPUs für sich). Phasen: `ref` (Basis: Tokens und Durchsatz mit allem auf
der Karte, daraus die Zahlen des Budgets), `stream` (Headroom so, dass `pct` % der Statik
streamen, Tokens müssen gleich sein, `bench.py` zeigt die Kosten), `status` (Manifest, Status und
Metriken zeigen den Anteil), `full` (Model `a` hält den VRAM, `b` aktiviert sich bei vollem VRAM,
streamt, antwortet gleich, `a` läuft weiter; Aktivierungszeit und erste Antwortzeit von `b`).
Die Herleitung der Headrooms steht im Skript; `m_ss.json` ist das Manifest der Phase `full` zum
Von-Hand-Start (`HA` ersetzen). Für `b` im Zustand `full`: `claimable_b = claimable_0 - used_a` mit
`used_a = F - 16 MiB - H_a` (F die freien MiB der Basis), also `H_a = F - 16 - claimable_0 + K + (1-pct)·S`
(K der Cache einer Sequenz, S die Statik, alles in MiB aus dem Startbericht der Basis).

## GPU-Messung (2× R9700, TP=2, PCIe 4.0 ×16)

`static_streaming.py … --seqs 8 --pct 25|50|100` (4096 Token, 8 Sequenzen; die Statik, die nicht
neben einen Cache für eine Sequenz passt, streamt). Anteil = `static_streamed_fraction` aus
`/radiance/models`; der Anteil ist eine Treppe (ganze Units), der Zielwert wird auf ±15 Punkte
getroffen. Greedy-Ausgabe (sechs Prompts, Text und Token) in allen Läufen **bit-identisch** zur
Referenz: `stream`, `status` (Manifest), `full` (Aktivierung bei vollem VRAM). Vollständige Tabellen,
Fehler und Fixes: `gpu-acceptance.md` §10.

| Model | Anteil gestreamt | C1 tok/s | C8 tok/s | Prefill tok/s (TTFT ms), 3,6k Token |
|---|---|---|---|---|
| Qwen3.5-0.8B (dicht) | 0 % (Referenz) | 323–330 | 1970–2001 | 67 500–68 600 (53) |
| | 25,6 % | 198–199 | 1337–1345 | 60 800–61 700 (59–61) |
| | 50,8 % | 62,6–62,9 | 474–476 | 42 200–43 200 (84–86) |
| | 85,5 % (Obergrenze mit 8 Sequenzen) | 32,4–32,5 | 226–247 | 31 500–34 500 (105–115) |
| | 99,998 % (1 Sequenz) | 25,6 | 23,5–25,5 (1 Seq.) | 26 000–26 700 (136–140) |
| Flash-Next-MoE | 0 % (Referenz, Experts teils im Pool) | 139–181 | 569–656 | 5715–6279 (579–631) |
| | 15,4 % | 36,5–37,4 | 61–74 | 2713–2846 (1273–1327) |
| | 42,7 % | 23,5–25,0 | 48–70 | 2245–2485 (1447–1590) |
| | 96,5 % | 11,2–11,5 | 42–65 | 2067–2212 (1665–1755) |

Decode wird um die gelesenen Bytes langsamer (dicht, 50 %: +12,9 ms je Token bei 969 MB
gestreamter Statik über beide Ranks; aus den Messungen nicht auf eine Link-Bandbreite zurückgerechnet).
Der Prefill kostet über den Ring viel weniger als Decode (0,8B, 85 %: −50 %
gegen −90 % im Decode). Beim MoE kommt zum Streaming der Statik dazu, dass die Karte keinen Platz
mehr für Experts hat: sie liegen alle im Pool, die Kosten sind also beides, nicht allein die Statik.

Aktivierung bei vollem VRAM (Model `a` hält die Karten, `b` aktiviert sich): dicht 2,1–2,3 s
Aktivierung, 2,2–2,6 s bis zur ersten Antwort; MoE 35,5–37,0 s, 36,2–37,7 s (Gewichte aus der
Platte, `--ngram-placement disk`). `a` läuft danach unverändert weiter.

## Zurück in den VRAM (#19)

Gestreamte Statik ist nur eine Übergangslösung (ADR 0002). Sobald die Karte Platz hat, wandert sie in
den VRAM: weil ein anderes Model deaktiviert wurde, ein elastischer Expert-Slab Seiten abgab (#35) oder
ein Cache Platz hergab. Das Signal ist der **physisch freie VRAM der Karte**, nicht ein logisches
Angebot: nur Bytes, die die Karte wirklich frei hat, werden vergeben.

| Stelle | Änderung |
|---|---|
| `core/place/planner.*` | `static_keep_order(prog, plan)`: die Reihenfolge, in der `assign_static_stream` die Karte füllt (Drafter, modellweite Gewichte, Schichten in Ausführungsreihenfolge, Türme, Embedding), jetzt eine Funktion. Der Mover holt die gestreamten Units in derselben Reihenfolge zurück: das Sicherste zuerst. Kein zweiter Algorithmus, keine zweite Schlüsselliste. |
| `core/place/mover.*` | `Mover::promote_statics(cap)`: kopiert je Aufruf einen Schub (bis 256 MiB) gestreamter statischer Units aus ihrem Pool-Slot in frischen VRAM und veröffentlicht die neue Adresse. Der Platz kommt aus dem Adressraum des elastischen Gewichte-Pools (`Pool::va_arena`, nie zweimal gemappt, `Pool::commit`; siehe `elastic-slab.md`, Aliasing!), die Kopie läuft als DMA-Liste (`RAD_COPY_YIELD`) auf dem Mover-Stream. Eine Unit, die nicht in `cap` passt, wird übersprungen, kleinere dahinter dürfen. Eine abgelehnte Commit-Anfrage oder weniger als `grow_floor` (96 MiB) frei lässt den Schub ausfallen und es wird später erneut versucht (250 ms bzw. 1 s). Die Pool-Kopie bleibt gehalten: jeder Schritt, der vor der Veröffentlichung ausgegeben wurde, liest weiter gültige Bytes, und die Veröffentlichung ist der normale Tabellenabgleich am Anfang des nächsten Schritts (`Ctx::sync_residency`, derselbe Weg wie beim Stager). Eine Unit mit laufender Kopie oder im VRAM stagt der Ring nicht mehr (`Mover::stageable`). |
| `core/place/group_planner.*` | `StaticFacts` je Rang (was das Zurückholen aller noch gestreamten Units kostet, was schon gehalten wird, wie beschäftigt das Model ist), `update_free` (freie Bytes der Karte), `group_statics`: je Rang gehen die freien Bytes abzüglich der Zusagen, die noch nicht genutzt wurden (Statik und Experts), an das beschäftigteste Model zuerst, je höchstens bis zu dem, was es noch streamt (`RankBudget::statics_cap`). Eine Zusage bleibt reserviert, bis das Model sie hält: zwei Nehmende bekommen nie dieselben Bytes. Gehaltenes wird nie zurückverlangt. |
| Vorrang vor Experts | Statik wird jeden Token gelesen, ein Expert nur, wenn er geroutet wird: die Statik-Zusagen kommen von den freien Bytes ab, bevor `group_gate` einem elastischen Model Wachstum erlaubt (`static_pending`). Eine Expert-Zusage, die schon gemacht und noch nicht gemappt ist, behält ihre Bytes. Die Reihenfolge unter mehreren Statik-Nehmern: Auslastung (`StaticFacts::heat`: 1 solange das Model läuft, abklingend im Leerlauf wie die Expert-Heat), dann Eintrittsreihenfolge. |
| `core/engine.cpp` | `Engine::group_tick` (alle 100 ms, zwischen Steps und im Leerlauf, wenn die Rank-Threads geparkt sind): meldet freie Bytes und Statik-Zahlen, liest das Ziel und ruft `promote_statics`. Laufende Requests bleiben ungestört: die Adressen ändern sich nur an Step-Grenzen. |
| `core/engine_bringup.cpp` | Ein Model in einer Gruppe, das Statik streamt, bekommt den elastischen Gewichte-Pool (8 TiB Adressraum, `weights_va_extra_`) auch ohne Experts. Ohne Manifest-Betrieb mit mehr als einem Model, ohne VMem oder unter `RADIANCE_ELASTIC_SLAB=off` bleibt alles wie in #18 (kein Zurückholen). |
| Status | `/radiance/models` und `/metrics` zeigen den **aktuellen** Anteil (`static_streamed_*`): der GroupPlanner führt die Zahlen, die die Engine meldet (Summe über die Ränge), sonst gilt der Plan. Der Anteil fällt sichtbar auf 0. Der Mover-Report nennt `static weights: N units promoted into VRAM`, das Log je Schub `promoted static weights into VRAM; X of Y still read from the pinned host pool`. |

Grenzen:

- Was einmal im VRAM ist, bleibt dort bis zur Deaktivierung des Models. Wird später ein anderes Model
  aktiviert, sieht es weniger freien VRAM; reicht der nicht, greift der Druckpfad (`pressure-deactivation.md`:
  das am längsten idle Model wird deaktiviert) oder das neue Model streamt selbst. Ein Zurückgeben an ein
  anderes Model (Demote der Statik) gibt es nicht.
- Pro Rang wird unabhängig entschieden (die Statik ist rangweise ein Shard und kein Kernel hängt davon ab, wo
  die Bytes eines Shards liegen); der Anteil in Status und Metriken ist die Summe.
- Der Adressraum ist endlich (8 TiB je Rang, eine Unit je Adresse, nie wiederverwendet): ein Zurückholen aller
  Units je Aktivierung braucht ein paar GiB davon.
- `--deterministic` streamt nicht (#18) und holt deshalb auch nichts zurück.
- Der Prefill-Ring bleibt in der Arena, wo er war; ohne gestreamte Units stagt er nichts mehr.

### Tests (Host, `ROCR_VISIBLE_DEVICES=99`)

- `place_test`: `static_keep_order` ist die Reihenfolge, in der der Planner die Karte füllt; die Statik
  wird in Reihenfolge, in frischen Arena-Adressen, byte-genau und höchstens bis `cap` zurückgeholt (eine Unit, die
  nicht passt, wartet), der Stager kopiert sie danach nicht mehr; eine Karte ohne Platz (weniger als `grow_floor`
  frei) und ein einfacher (nicht elastischer) Pool holen nichts zurück.
- `group_planner_test`: die freien Bytes gehen an das beschäftigteste Model zuerst und nie über das hinaus, was es
  streamt; eine ungenutzte Zusage bleibt reserviert und nichts Gehaltenes wird zurückverlangt (kein Doppelverplanen);
  Statik nimmt die freien Bytes vor einem Expert-Wachstum; ein Rang entscheidet für sich (TP=2).
- `engine_e2e_test` (libref und libavx, `RADIANCE_HOST_VRAM_MIB`): ein Manifest mit `a` und `b` (dasselbe
  `g31`); `a` lässt `b` Platz für einen Teil seiner Statik (der Headroom von `a` wird gesucht, nicht fest), `b`
  streamt 65,9 %; `a` wird im Leerlauf deaktiviert und `b`s Anteil fällt auf 0, `static_streamed_bytes` 0; jede
  Antwort von `b` davor, dabei (16 bzw. 43 Anfragen am Stück) und danach ist Token für Token die der Referenz (alles
  auf der Karte, `b` allein).

- `model_deactivation_test`: Status und Metriken folgen den Zahlen, die die Engine dem GroupPlanner meldet (der
  gestreamte Anteil fällt auf 0), sonst gilt der Plan.

## GPU-Messung (#19)

Siehe `gpu-acceptance.md` §11. Qwen3.5-0.8B (dicht), 2× R9700, TP=2, `--seqs 8`: `b` streamt, `a` wird im
Leerlauf deaktiviert, der Anteil fällt innerhalb von 1–2 s nach der Deaktivierung auf 0, die Ausgabe bleibt
bit-identisch (Greedy, `gen.py`), und der Durchsatz erholt sich auf das Niveau ohne Streaming:

| gestreamter Anteil von `b` | C1 tok/s | C8 tok/s | Prefill tok/s (TTFT ms) |
|---|---|---|---|
| 83,9 % (Obergrenze, `a` aktiv) | 32,1 | 252,8 | 33 400 (109) |
| nach dem Zurückholen (0 %) | 331,7 | 2016,9 | 62 700 (59) |
| 32,7 % | 108,7 | 771,9 | 49 400 (74) |
| nach dem Zurückholen (0 %) | 330,2 | 2013,1 | 66 800 (55) |
| 6,4 % | 301,5 | 1862,4 | 67 000 (55) |
| nach dem Zurückholen (0 %) | 331,3 | 2017,6 | 66 500 (56) |
| Referenz: ein Model, alles auf der Karte | 333,3 | 2053,3 | 70 000 (52) |
