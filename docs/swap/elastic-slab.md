# Elastischer Expert-Slab (#35)

Begriffe: `GLOSSARY.md`. Vorgänger: `slot-release.md` (#15, logische Abgabe), `kv-lend.md` (#17).

## Was sich ändert

In Manifest-Betrieb mit mehr als einem Model und einem MoE mit Experts liegt der Gewichte-Pool einer
Engine auf VMem (`Pools::configure(..., weights_va_extra)`, `Pool::init_elastic`). Der Pool reserviert
frischen Adressraum über sein Budget hinaus (8 TiB je Rank, nur virtuell, `Pool::va_arena`). Der Slab einer
Klasse wird in Stücken von etwa 128 MiB (`MoverConfig::slab_chunk_bytes`, `Mover::pick_chunk`) auf je eigenem
Arena-Adressraum gemappt und stückweise wieder entmappt; ein Stück, das zurückgegeben wurde, bekommt beim
nächsten Wachstum eine neue Adresse (`ClassPools::slot_ptr`). Eine Adresse wird nie ein zweites Mal gemappt. Ohne Manifest, mit einem Model, ohne
Experts oder ohne VMem bleibt der Pool eine einzige `rad_dev_alloc` wie bisher (Einzel-Engine
unverändert, siehe Messung in `gpu-acceptance.md`). Ausnahme (#19): ein Model in einer Gruppe, das statische
Gewichte streamt, bekommt den elastischen Pool auch ohne Experts; die Arena nimmt dort die Statik auf, die
`Mover::promote_statics` in den VRAM zurückholt (`static-streaming.md`), ebenfalls auf frischen Adressen und ohne je
etwas zu entmappen.

- **Mover** (`Mover::set_resident_cap`, `resize`): Ein Shrink gibt die obersten Slots logisch ab (wie #15)
  und `shrink_backing` entmappt das oberste Stück (`Pool::decommit`), sobald kein Slot darin mehr belegt oder
  vom Segment abgedeckt ist (nach Retire/Quarantäne); die Freigabe geschieht stückweise, ein Aufruf je Stück.
  Ein Regrow über den Plan hinaus mappt die nächsten Slots (`grow_backing`, `map_slots`, in Schritten von
  128 MiB, ein neues Stück auf neuem Adressraum) und hebt das Segment erst über gebackte Slots.
  Eine abgelehnte Commit-Anfrage (Karte voll oder weniger als `grow_floor` frei) stoppt das Wachstum und
  versucht es später erneut. Tabellen werden nur über die normalen Moves veröffentlicht.
- **GroupPlanner** (`RankFacts::elastic/held`, `group_gate`): Der Pool ist die Summe der Angebote. Ein
  elastisches Model darf bis zum ganzen Pool zugeteilt bekommen, ein Wachstum wird aber nur aus Bytes
  gezahlt, die kein Mitglied hält (`Pool − Σ held`). `held` meldet die Engine nach dem Entmappen
  (`update_held`), erst dann steigt das Target des Nehmenden. Ein noch nicht genutzter Grant bleibt
  reserviert, zwei Nehmende bekommen nie dieselben freien Bytes. Gaben (Target < held) gelten sofort.
- **Klassen im Gleichschritt**: Die zwei Hälften eines Experts (gate_up/down) liegen in zwei Slab-Klassen.
  Ein elastisches Model verteilt seine Bytes proportional zu den Planes statt nach Klassen-Heat, sonst
  wächst die eine Klasse, während die andere schrumpft (Commits und Kopien ohne Nutzen).
- **Idle-Heat kühlt ab** (`Engine::group_tick`): Der Heat-Zähler zerfällt je Step. Eine Engine ohne
  Requests meldet ihre Heat mit Halbwertszeit 4 s und nach 20 s als kalt; sie behält nur ihr Minimum.
  Zielwerte unter 1 % Änderung werden nicht angewendet.
- **Messwerte**: `/radiance/models` (`expert_slab`: je Rank `committed_bytes`, `target_bytes`,
  `offer_bytes`) und `/metrics` (`radiance:model_expert_slab_committed_bytes`, `…_target_bytes`).
  `committed` zählt die beweglichen Expert-Slots mit Seiten dahinter (ohne Arbeitsreserve und feste
  Residents); `offer` ist dieselbe Zahl, wie der Plan sie platziert hat (die Engine meldet sie beim
  ersten Tick und ersetzt damit das Angebot des Resolvers, der auch die festen Slots enthält).

## Grenzen

- Der KV-Carve wächst jetzt in Stücken auf frischem Adressraum, aus dem, was die Slabs zurückgeben
  (`kv-lend.md`, Teil 2 von #17). Der Slab selbst ist davon unverändert.
- Ein Model, das erst aktiviert wird, während ein anderes über sein Angebot hinaus gewachsen ist, startet
  mit dem, was frei ist; das Wachstum des anderen schrumpft binnen Sekunden zurück, wenn das Target fällt.
- Bis die Engine ihre erste Messung meldet, gilt das Angebot des Resolvers als belegt (`held = Angebot`),
  damit ein anderes Model in der Aktivierungsphase nicht in die Bytes wächst.
- Die Arena ist endlich (8 TiB, bei 128-MiB-Stücken rund 65 000 Stückwechsel je Rank). Ist sie verbraucht,
  hört das physische Entmappen und Mappen auf (Warnung „address arena is used up"); die Caps wirken dann
  nur noch auf den bereits gemappten Teil.
- `RADIANCE_ELASTIC_SLAB=off` lässt in einer Gruppe den einfachen Pool (Kontrolle für Messungen).
- `--host-pool-mib` bestimmt den Expert-Floor (`expert_floor`): ein Model kann nicht unter das schrumpfen,
  was sein Host-Pool nicht hält. Mit 20 000 MiB je Model (m_mm.json) liegt der Floor fast beim Angebot, es
  gibt dann nichts zu tauschen; die Messung nutzt 26 000 MiB (`m_el.json`).

## Tests

`tests/place_test.cpp` (`an_elastic_slab_*`, `a_plain_weights_pool_*`): Shrink gibt Seiten zurück, Regrow
über den Plan hinaus backt neue, Grenze durch den Adressraum, Daten bleiben byte-genau, ein einfacher Pool
bleibt unberührt. Die Host-Backend-VMem (mmap) ist die Fake-Device-Schicht; dieselben Tests laufen auf der
Karte (`place_test_card`). `tests/mem_test.cpp` (`reserve_growing`), `tests/group_planner_test.cpp`
(Elastic: Zuteilung über das Angebot, Grant erst nach Freigabe, keine doppelte Vergabe, Neuberechnung bei
Leave, Rank-weise Entscheidung, Gleichschritt der Klassen).

## Nicht endliche Logits: Ursache und Fix

**Befund.** Unter Last mit wechselndem Schrumpfen und Wachsen lieferte vereinzelt eine Anfrage mit `logprobs`
kein endliches Logit (`row 0 has no finite logit`, HTTP 500); ein Model blieb danach bis zu 30 s kaputt (bis die
fehlerhaften Residents getauscht waren). Ausgelöst wurde das reproduzierbar von außen: `rocm-smi --showmeminfo
vram` jede 2 s oder schon ein Programm, das die Render-Nodes `/dev/dri/renderD12x` öffnet und schließt, genügte.
Reine `cat /sys/class/drm/*/device/mem_info_vram_used`-Abfragen und `rsmi_init` allein nicht.

**Ursache.** `hsa_amd_vmem_map` auf eine Adresse, die schon einmal gemappt und wieder entmappt war, wirkt auf
diesem ROCr (HIP 7.x) und Kernel 7.0.2-6-pve nicht: die Adresse zeigt weiter auf die erste physische Seite. Hat
diese Seite inzwischen ein anderes Handle bekommen (genau das passiert, wenn ein Model Seiten abgibt und das andere
sie nimmt), lesen und schreiben zwei Adressen dieselbe Seite; die Daten eines Experts werden von einem anderen
überschrieben. Belegt ohne die Engine mit `tests/device_test.cpp` (`vmem_stress_*`, `RAD_VMEM_STRESS=<s>`): ein
Bereich von 256 Granulen wird 45 s lang gemappt, beschrieben, entmappt, mit Zufallsadressen:

| Variante | Lauf | falsche Granulen im geänderten Bereich |
|---|---|---|
| Adressen wiederverwendet, ohne fremden Client | 45 s, 1264 Wechsel | 8385 (und 0 direkt nach dem Schreiben) |
| Adressen wiederverwendet, mit fremdem Render-Node-Client | 45 s, 1248 Wechsel | 8256 |
| Adressen wiederverwendet, 50 ms Pause zwischen Unmap und Map | 30 s, 401 Wechsel | 1724 |
| Adressen wiederverwendet, Handle vor dem Unmap freigegeben | 30 s, 1257 Wechsel | 8331 |
| Handles in einem Pool wiederverwendet statt freigegeben | 30 s, 1249 Wechsel | 6117 |
| jede Adresse freigegeben und neu reserviert (Reserve liefert dieselbe Adresse) | 30 s, 1249 Wechsel | 6377 |
| Handle nie freigegeben (Speicher leckt, keine Seite wird wiederverwendet) | 30 s, 1217 Wechsel | 0 |
| **jede Adresse nur einmal gemappt** | 30 s, 1209 Wechsel | **0** |
| **jede Adresse nur einmal gemappt, mit fremdem Render-Node-Client** | 45 s, 1825 Wechsel; 20 s, 809 Wechsel | **0 / 0** |

Der falsche Inhalt ist das Muster einer späteren Schreibung auf eine andere Adresse (Aliasing, im Test
nachgewiesen), oder Nullen. Die unberührte Vergleichsfläche blieb immer unverändert. Ein Lesen einer entmappten
Adresse faultet nicht, es liefert andere Daten. Das Öffnen eines Render-Nodes durch einen fremden Prozess
verschlimmert es nur (die Karte ist voll und der Kernel belegt VRAM für dessen Seitentabellen, ein Eingriff in die
Mappings des Prozesses), die Wiederverwendung der Adresse reicht aus.

**Gegenproben in der Engine** (`scripts/swap-acceptance/lpstress.py`, Last abwechselnd auf m1 und m2, jede
Anfrage mit `logprobs`, danach je 12 Einzelanfragen an das ruhende Model; Fehler = HTTP 500):

| Aufbau | Zyklen | Anfragen | Fehler | Zyklen mit Fehler |
|---|---|---|---|---|
| ohne Beobachter (lp0, lp1, lp2) | 4 + 4 + 5 | 628 + 366 + 1058 | 0 | 0 |
| `rocm-smi`-Aufruf alle 2 s (lp3) | 5 | 1337 | 653 | 4 von 5 |
| derselbe, einfacher Pool in der Gruppe (lp4, `RADIANCE_ELASTIC_SLAB=off`) | 4 | 592 | 0 | 0 |
| `cat mem_info_vram_used` (lp5) | 4 | 715 | 0 | 0 |
| `rsmi_init` allein (lp6) | 4 | 708 | 0 | 0 |
| `rsmi_init` + Speicherabfrage (lp7) | 4 | 695 | 1 | 1 von 4 |
| Render-Nodes öffnen alle 2 s (lp8) | 4 | 1053 | 514 | 2 von 4 |
| dasselbe, einfacher Pool (e1) | 4 | 594 | 0 | 0 |
| dasselbe, elastischer Pool ohne Größenwechsel (e2) | 4 | 662 | 0 | 0 |
| dasselbe, 1100 MiB mehr Headroom bei m2 (h1) | 4 | 1404 | 778 | 2 von 4 |

Es braucht Größenwechsel (Unmap, Map derselben Adresse) plus Anstoß von außen. Ohne Beobachter ist die Rate klein, aber nicht null: die ersten Läufe mit dem alten
Stand (Stufen davor in `gpu-acceptance.md`, Abschnitt 9) hatten bei der einmaligen Prüfung nach den Phasen in
etwa der Hälfte der Läufe einen Fehler.

Ausgeschlossen wurden: ein Leser einer freien Slot-Adresse (die freigegebenen Slots mit 0xFF füllen statt
entmappen: kein Fehler), Stream-Synchronisation vor dem Entmappen, Entmappen nur im Leerlauf, Kopien über
Blit-Kernel statt der DMA-Liste, der Fehlerpfad eines abgelehnten Commits, zu wenig freier VRAM
(1100 MiB Headroom ändert nichts), Granulgröße gegen Slot-Stride (die Randgranulen bleiben gemappt), ein Peer-Zugriff
(der Slab wird nur vom eigenen Rank gelesen).

**Fix.** Der Slab ist nicht mehr ein zusammenhängender Bereich, der an derselben Adresse wieder gemappt wird, sondern
besteht aus Stücken auf je neuem Adressraum (`Pool::va_arena`, nie zweimal vergeben). Ein Stück wird als Ganzes
gemappt und als Ganzes entmappt; seine Adresse wird danach nicht mehr benutzt. Slot-Adressen werden über
`ClassPools::slot_ptr` aufgelöst und ändern sich, wenn ein Stück zurückkommt; das ist sicher, weil nur freie Slots
(ohne Unit, ohne Kopie in Flug, nach der Quarantäne) im entmappten Stück liegen und Tabellen nur über die
normalen Moves veröffentlicht werden. Die Stückgröße (`pick_chunk`) wird so gewählt, dass Stück und Tail-Pad ohne
Verschnitt in ganze Granulen passen. Kosten: das Zurückgeben geschieht in Stücken von rund 128 MiB statt exakt;
Ein Stück, das teilweise gebraucht wird, bleibt gemappt.

**Nach dem Fix** (gleiche Skripte, mit den fremden Beobachtern gleichzeitig; `smi` = `rocm-smi`-Aufruf alle 2 s,
`drm` = Render-Nodes öffnen alle 2 s):

| Lauf | Beobachter | Zyklen | Anfragen | Fehler |
|---|---|---|---|---|
| c1 | drm | 5 | 962 | 0 |
| c2 | drm | 5 | 963 | 0 |
| L1 | smi + drm | 10 | 1880 | 0 |
| L2 | smi + drm | 10 | 1971 | 0 |
| zusammen | | 30 | 5776 | **0** |

Das Karten-Unit-Test-Binary (`place_test -perf_`) und `vmem_stress_*` mit `RAD_VMEM_STRESS_FRESH=1` (809 und 1825
Wechsel, mit fremdem Client) bleiben ebenfalls fehlerfrei.
