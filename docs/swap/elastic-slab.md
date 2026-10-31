# Elastischer Expert-Slab (#35)

Begriffe: `GLOSSARY.md`. Vorgänger: `slot-release.md` (#15, logische Abgabe), `kv-lend.md` (#17).

## Was sich ändert

In Manifest-Betrieb mit mehr als einem Model und einem MoE mit Experts liegt der Gewichte-Pool einer
Engine auf VMem (`Pools::configure(..., weights_va_extra)`, `Pool::init_elastic`). Der Pool reserviert
Adressraum über sein Budget hinaus (Expert-Plane × 1,25 + 64 MiB); `Pool::reserve_growing` hängt hinter
jeden Slab einen Adress-Schwanz an, der nicht aufs Budget zählt. Ohne Manifest, mit einem Model, ohne
Experts oder ohne VMem bleibt der Pool eine einzige `rad_dev_alloc` wie bisher (Einzel-Engine
unverändert, siehe Messung in `gpu-acceptance.md`).

- **Mover** (`Mover::set_resident_cap`, `resize`): Ein Shrink gibt die obersten Slots logisch ab (wie #15)
  und `shrink_backing` entmappt die Seiten darüber (`Pool::decommit`), sobald kein Slot dort mehr belegt
  ist (nach Retire/Quarantäne), 256 MiB je Aufruf. Ein Regrow über den Plan hinaus committet die Seiten
  des Schwanzes in 128-MiB-Schritten (`grow_backing`) und hebt das Segment erst über gebackte Slots.
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

- Der KV-Carve bleibt fest (`KvPartition::growable` = 0); die modellübergreifende KV-Leihe ist unverändert.
- Ein Model, das erst aktiviert wird, während ein anderes über sein Angebot hinaus gewachsen ist, startet
  mit dem, was frei ist; das Wachstum des anderen schrumpft binnen Sekunden zurück, wenn das Target fällt.
- Bis die Engine ihre erste Messung meldet, gilt das Angebot des Resolvers als belegt (`held = Angebot`),
  damit ein anderes Model in der Aktivierungsphase nicht in die Bytes wächst.
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
